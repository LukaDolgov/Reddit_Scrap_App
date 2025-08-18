# reco_ui.py
import os
import re
import math
import time
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------------------
# Service account / secrets setup (your original logic)
# ---------------------------------------
sa_secret = None
if st.secrets.get("google") and st.secrets["google"].get("service_account_key"):
    sa_secret = st.secrets["google"]["service_account_key"]

# Also support a flat key name for convenience
if not sa_secret and st.secrets.get("GCP_SA_JSON"):
    sa_secret = st.secrets["GCP_SA_JSON"]

if sa_secret:
    keyfile = "/tmp/reddit-scrapper-sa.json"
    with open(keyfile, "w", encoding="utf-8") as f:
        f.write(sa_secret)
    try:
        os.chmod(keyfile, 0o600)
    except Exception:
        # Some platforms may not support chmod; ignore if it fails
        pass
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keyfile

# ======= Environment variables (from secrets with local fallback) =======
# Project & location — prefer explicit secret keys if present
os.environ.setdefault(
    "GOOGLE_CLOUD_PROJECT",
    st.secrets.get("google", {}).get("project_id", os.environ.get("GOOGLE_CLOUD_PROJECT", "reddit-scrapper-468019"))
)
os.environ.setdefault(
    "GOOGLE_CLOUD_LOCATION",
    st.secrets.get("google", {}).get("location", os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
)

# Reddit creds — prefer secrets, then environment variables for local dev
CLIENT_ID = st.secrets.get("reddit", {}).get("REDDIT_CLIENT_ID") or os.environ.get("REDDIT_CLIENT_ID")
CLIENT_SECRET = st.secrets.get("reddit", {}).get("REDDIT_CLIENT_SECRET") or os.environ.get("REDDIT_CLIENT_SECRET")
REFRESH_TOKEN = st.secrets.get("reddit", {}).get("REDDIT_REFRESH_TOKEN") or os.environ.get("REDDIT_REFRESH_TOKEN")
USER_AGENT = st.secrets.get("reddit", {}).get("REDDIT_USER_AGENT") or os.environ.get("REDDIT_USER_AGENT", "reddit-scrap/0.1")

# Basic checks / user-visible debug (remove when you are done testing)
st.info("Initializing credentials...")
try:
    import google.auth
    creds, proj = google.auth.default()
    st.write("Project detected:", proj)
    st.write("Service account (if any):", getattr(creds, "service_account_email", type(creds).__name__))
    st.write("Location:", os.environ.get("GOOGLE_CLOUD_LOCATION"))
except Exception as e:
    st.warning("google.auth.default() failed: " + str(e))

# ---------------------------------------
# Try to reuse your existing functions. If not found, minimal fallbacks are used.
# ---------------------------------------
HAS_SENTIMENT_MODULE = False
try:
    from sentiment import get_latest_submission, classify_comments_with_gemini, safe_replace_more
    HAS_SENTIMENT_MODULE = True
except Exception:
    # Fallback placeholders (keeps app working even if sentiment.py isn't available)
    def get_latest_submission(username: str):
        raise RuntimeError("Please import your get_latest_submission from your sentiment.py into the PYTHONPATH.")

    def classify_comments_with_gemini(comments: List[str]):
        # naive fallback: mark all as HUMAN (no Gemini)
        return [{"index": i + 1, "label": "HUMAN", "reason": ""} for i in range(len(comments))]

    # robust fallback implementation for safe_replace_more using prawcore if available
    def safe_replace_more(comments_object, limit):
        """Fallback safe_replace_more that handles 429s by backing off."""
        try:
            import prawcore
        except Exception:
            prawcore = None

        backoff = 1
        while True:
            try:
                comments_object.replace_more(limit=limit)
                return
            except Exception as e:
                # If it's a TooManyRequests from prawcore, honor retry_after or exponential backoff
                if prawcore and isinstance(e, prawcore.exceptions.TooManyRequests):
                    wait = getattr(e, "retry_after", backoff)
                    st.write(f"429 TooManyRequests: sleeping {wait}s")
                    time.sleep(wait)
                    backoff = min(backoff * 2, 60)
                    continue
                # For other exceptions, re-raise so upstream can handle/log
                raise

# ---------------------------------------
# Helper utilities (sentiment, vwap, sma, atr, kelly, relevance)
# ---------------------------------------

# Keep your original simple sentiment keywords
BULL_KEYWORDS = {"buy", "bull", "moon", "long", "pump", "call", "rocket", "tendies", "alpha", "hodl"}
BEAR_KEYWORDS = {"sell", "bear", "short", "dump", "exit", "weak", "stop", "down"}

def simple_sentiment_score(comments: List[str]):
    """Return (n_bull, n_bear, n_total, score_norm) for list of comment strings."""
    n_bull = n_bear = 0
    for c in comments:
        text = (c or "").lower()
        for k in BULL_KEYWORDS:
            if k in text:
                n_bull += 1
                break
        for k in BEAR_KEYWORDS:
            if k in text:
                n_bear += 1
                break
    n_total = len(comments)
    score = (n_bull - n_bear) / max(1, n_total)   # in [-1,1]
    return n_bull, n_bear, n_total, score

def compute_vwap(df: pd.DataFrame):
    """Compute VWAP for dataframe with columns ['High','Low','Close','Volume']"""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    return pv.cumsum() / df["Volume"].cumsum()

def sma(series: pd.Series, window: int):
    return series.rolling(window).mean()

def fractional_kelly(p: float, b: float, fraction: float = 0.5) -> float:
    """Return position fraction of bankroll using fractional Kelly."""
    q = 1 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return max(0.0, f_full) * fraction

# New: relevance filter (so we can filter comments by ticker/company BEFORE classification)
def filter_comments_by_relevance(comments: List[str], ticker: str, company: str, aliases: List[str] = None) -> List[str]:
    """Simple regex-based relevance filtering; returns comments likely about the ticker/company."""
    aliases = aliases or []
    ticker_low = (ticker or "").lower()
    company_low = (company or "").lower()

    def is_relevant(text: str) -> bool:
        t = (text or "").lower()
        # cashtag
        if ticker_low and re.search(r'\$' + re.escape(ticker_low), t):
            return True
        # whole-word ticker
        if ticker_low and re.search(r'\b' + re.escape(ticker_low) + r'\b', t):
            return True
        # company name
        if company_low and company_low in t:
            return True
        # aliases
        for a in aliases:
            if a and a.lower() in t:
                return True
        # context phrase: ticker/company + finance words
        if ticker_low or company_low:
            pattern = r'\b(' + re.escape(ticker_low or "") + '|' + re.escape(company_low or "") + r')\b.*\b(price|buy|sell|shares|stock|short|long|earnings|dividend|guidance|ipo|merger|hodl|sats|pump|dump|fud|moon)\b'
            if re.search(pattern, t):
                return True
        return False

    return [c for c in comments if is_relevant(c)]

# Utility to fetch up to N comments (keeps your original logic)
def get_up_to_n_comments(submission, max_comments=500, batch=25, pause_between_batches=0.5):
    """
    Return up to max_comments Comment objects from `submission`.
    - batch: how many 'MoreComments' placeholders to expand per iteration
    - pause_between_batches: small sleep to avoid blasting Reddit
    """
    # 1) Make sure we have the initial page of top-level comments (no deep expansion)
    safe_replace_more(submission.comments, limit=0)
    comments = submission.comments.list()
    if len(comments) >= max_comments:
        return comments[:max_comments]

    # 2) Iteratively expand more placeholders in batches until we have enough or nothing more to fetch
    while len(comments) < max_comments:
        prev_len = len(comments)
        try:
            safe_replace_more(submission.comments, limit=batch)
        except Exception as e:
            # If safe_replace_more raises for any other reason, bail gracefully
            st.write("Error expanding more comments:", e)
            break

        # small polite pause so we don't burn rate limits
        if pause_between_batches:
            time.sleep(pause_between_batches)

        comments = submission.comments.list()
        # if calling replace_more didn't increase the number of comments, we've exhausted reachable comments
        if len(comments) == prev_len:
            break

    return comments[:max_comments]

# ---------------------------------------
# CRYPTO ALIASES (include corn 🌽 for BTC)
# ---------------------------------------
CRYPTO_TICKER_MAP = {
    "BTC": {"pair": "BTC-USD", "name": "bitcoin", "aliases": ["btc", "bitcoin", "$btc", "sats", "hodl", "corn", "🌽"]},
    "SOL": {"pair": "SOL-USD", "name": "solana", "aliases": ["sol", "solana", "$sol"]},
}

# ---------------------------------------
# Streamlit UI (keeps your original layout, adds small sidebar options)
# ---------------------------------------
st.set_page_config(page_title="Reddit→Market Recs", layout="wide")
st.title("Reddit-driven Stock Recommendation (experimental)")
st.markdown(
    """
**Not financial advice.** This is an experimental tool that combines Reddit comment sentiment and simple technical rules to produce *suggestions* only. 
Always do your own research. (See: legal disclaimer.)  
"""
)

# Sidebar inputs (kept from your original file) + added controls
st.sidebar.header("Inputs")
username = st.sidebar.text_input("Reddit user to observe", value="wsbapp")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", value="AAPL")
# NEW: optional company name to help relevance filtering (left blank by default)
company_name = st.sidebar.text_input("Company name (for relevance, optional)", value="")
use_yf = st.sidebar.checkbox("Auto-fetch market data with yfinance", value=True)
current_price_input = st.sidebar.number_input("Current price (override)", value=0.0, format="%.4f")
account_size = st.sidebar.number_input("Account size (USD)", value=1000.0, step=100.0)
reward_risk = st.sidebar.number_input("Expected reward:risk (b)", value=1.5, step=0.1)
kelly_fraction = st.sidebar.slider("Kelly fraction", min_value=0.1, max_value=1.0, value=0.5)
sma_short_window = st.sidebar.number_input("SMA short window (days)", value=10, step=1)
sma_long_window = st.sidebar.number_input("SMA long window (days)", value=50, step=1)

# NEW: should we filter comments by ticker/company BEFORE classification?
filter_before_classify = st.sidebar.checkbox("Filter by ticker/company before classification (recommended)", value=True)
# NEW: show debug samples of relevant/non-relevant comments to tune filter
show_relevance_debug = st.sidebar.checkbox("Show relevance debug examples", value=False)

run = st.sidebar.button("Analyze now")

# Small helper to print debug examples
def debug_relevance_examples(all_comments, relevant_comments, n=6):
    st.write("**Sample relevant (up to n):**")
    for c in relevant_comments[:n]:
        st.write("-", c[:300])
    st.write("**Sample non-relevant (up to n):**")
    nonrel = [c for c in all_comments if c not in relevant_comments]
    for c in nonrel[:n]:
        st.write("-", c[:300])

# ---------------------------------------
# Main run flow (your original logic but stock path pre-filters if toggled)
# ---------------------------------------
if run:
    st.info("Fetching Reddit thread and comments...")
    try:
        submission = get_latest_submission(username)
    except Exception as e:
        st.error(f"Unable to fetch Reddit submissions: {e}")
        st.stop()

    st.markdown(f"**Latest post:** {submission.title}")
    # get top-level comments (your get_latest_submission should have done replace_more(limit=0))
    submission.comment_sort = "best"
    top_level = get_up_to_n_comments(submission, max_comments=400, batch=40, pause_between_batches=1.0)
    comments_texts = [c.body.replace("\n", " ") for c in top_level]

    st.write(f"Collected {len(comments_texts)} top-level comments (raw)")

    # Decide whether to filter by ticker/company before classification
    if filter_before_classify:
        # Use provided company_name when present, otherwise fall back to ticker
        company_for_relevance = company_name if company_name else ticker
        aliases = [ticker.lower()]
        if company_for_relevance and company_for_relevance.lower() != ticker.lower():
            aliases.append(company_for_relevance.lower())
        # For short tickers it's helpful to also add some common tokens (you can expand)
        if ticker.lower() == "sol":
            aliases.extend(["sol", "solana"])
        if ticker.lower() == "btc":
            aliases.extend(["btc", "bitcoin", "corn", "🌽", "hodl", "sats"])

        relevant = filter_comments_by_relevance(comments_texts, ticker, company_for_relevance, aliases=aliases)
        st.write(f"Relevant comments by ticker/company: {len(relevant)} out of {len(comments_texts)}")

        if show_relevance_debug:
            debug_relevance_examples(comments_texts, relevant, n=6)

        # CLASSIFY only the relevant ones with Gemini (saves calls & avoids noise)
        st.info("Detecting bot comments (Gemini) on relevant subset...")
        try:
            classifications = classify_comments_with_gemini(relevant)
        except Exception as e:
            st.warning("Gemini classification failed — proceeding without bot filtering.")
            classifications = [{"index": i + 1, "label": "HUMAN", "reason": ""} for i in range(len(relevant))]

        human_comments = []
        for i, txt in enumerate(relevant):
            lab = classifications[i] if i < len(classifications) else {"label": "HUMAN"}
            if isinstance(lab, dict):
                label = lab.get("label", "HUMAN")
            else:
                # if Gemini returned a JSON list or string, be defensive
                label = str(lab).upper()
            if "BOT" not in label:
                human_comments.append(txt)

        st.write(f"{len(human_comments)} comments classified as human & relevant.")
    else:
        # No pre-filter; classify all comments as before
        st.info("Detecting bot comments (Gemini) on full comment set...")
        try:
            classifications = classify_comments_with_gemini(comments_texts)
        except Exception as e:
            st.warning("Gemini classification failed — proceeding without bot filtering.")
            classifications = [{"index": i + 1, "label": "HUMAN", "reason": ""} for i in range(len(comments_texts))]

        human_comments = []
        for i, txt in enumerate(comments_texts):
            lab = classifications[i] if i < len(classifications) else {"label": "HUMAN"}
            if isinstance(lab, dict):
                label = lab.get("label", "HUMAN")
            else:
                label = str(lab).upper()
            if "BOT" not in label:
                human_comments.append(txt)

        st.write(f"Collected {len(comments_texts)} top-level comments — {len(human_comments)} classified as human.")

    # sentiment
    n_bull, n_bear, n_tot, score = simple_sentiment_score(human_comments)
    st.metric("Sentiment score (normalised)", f"{score:.3f}")
    st.write(f"Bullish: {n_bull}, Bearish: {n_bear}, Total human comments: {n_tot}")

    # Estimate p (probability of a 'win') from sentiment: map score [-1,1] -> p in [0.05,0.95]
    p_est = 0.5 + 0.4 * score   # center 0.5, move by up to 0.4
    p_est = max(0.05, min(0.95, p_est))

    # Market data (optional)
    hist = None
    if use_yf:
        st.info("Fetching market data from Yahoo Finance...")
        try:
            df = yf.Ticker(ticker).history(period="90d", interval="1d")
            if df.empty:
                st.warning("No market data returned for ticker.")
            else:
                df = df.rename(columns={c: c.capitalize() for c in df.columns})
                df = df[["High", "Low", "Close", "Volume"]]
                df["VWAP"] = compute_vwap(df)
                df["SMA_short"] = sma(df["Close"], sma_short_window)
                df["SMA_long"] = sma(df["Close"], sma_long_window)
                current_price = current_price_input if current_price_input > 0 else df["Close"].iloc[-1]
                hist = df
        except Exception as e:
            st.warning("yfinance fetch failed: " + str(e))
            current_price = current_price_input if current_price_input > 0 else 0.0
    else:
        current_price = current_price_input if current_price_input > 0 else 0.0

    st.write(f"Using price = {current_price:.4f}")

    # Compute Kelly-based position fraction
    pos_frac = fractional_kelly(p=p_est, b=reward_risk, fraction=kelly_fraction)
    pos_usd = pos_frac * account_size

    st.subheader("Suggested position sizing (Kelly-based)")
    st.write(f"Estimated win probability p = {p_est:.2f}")
    st.write(f"Reward:risk b = {reward_risk:.2f}")
    st.write(f"Fraction of bankroll (fractional Kelly) = {pos_frac:.4f}")
    st.write(f"Suggested position size = ${pos_usd:,.2f} (on account ${account_size:,.0f})")

    # Entry/exit logic (simple rule-based)
    decision = "HOLD"
    entry_price = None
    stop_loss = None
    take_profit = None

    if hist is not None and current_price > 0:
        sma_short = hist["SMA_short"].iloc[-1]
        sma_long = hist["SMA_long"].iloc[-1]
        vwap = hist["VWAP"].iloc[-1]
        st.markdown("### Technical snapshot")
        st.write(f"SMA({sma_short_window}) = {sma_short:.4f}  |  SMA({sma_long_window}) = {sma_long:.4f}  |  VWAP = {vwap:.4f}")

        # rule example: bullish sentiment + price above SMA short -> momentum buy
        if score > 0.15 and current_price > sma_short:
            decision = "BUY (momentum)"
        # contrarian buy: high bullish sentiment but dip below VWAP -> buy the dip
        elif score > 0.2 and current_price < vwap:
            decision = "BUY (dip)"
        elif score < -0.2 and current_price < sma_short:
            decision = "SELL (weak)"

        # compute entry/SL/TP around current price using reward:risk
        # default stop-loss risk fraction r (user can change)
        r = 0.03  # risk 3% default
        entry_price = current_price
        stop_loss = entry_price * (1 - r)
        take_profit = entry_price * (1 + r * reward_risk)
    else:
        # fallback: use sentiment-only rule
        if score > 0.2:
            decision = "CONSIDER BUY"
            entry_price = current_price
            stop_loss = entry_price * 0.97
            take_profit = entry_price * 1.05
        elif score < -0.2:
            decision = "CONSIDER SELL"

    st.subheader("Recommendation")
    st.write("Decision:", decision)
    if entry_price:
        st.write(f"Entry price: {entry_price:.4f}")
        st.write(f"Stop loss: {stop_loss:.4f}")
        st.write(f"Take profit: {take_profit:.4f}")

    st.markdown("---")
    st.caption(
        "This tool is experimental and educational. It combines social sentiment (Reddit) with simple technical indicators and math (Kelly) to produce suggestions — *not investment advice*. "
    )
