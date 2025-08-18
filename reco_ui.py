# reco_ui.py
import os
import re
import math
import time
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List

# ----- Service account / secrets setup (keep your existing logic) -----
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


# ----- Try to reuse existing functions from sentiment.py if present -----
HAS_SENTIMENT_MODULE = False
try:
    from sentiment import get_latest_submission, classify_comments_with_gemini, safe_replace_more
    HAS_SENTIMENT_MODULE = True
except Exception:
    # fallback placeholders
    def get_latest_submission(username: str):
        raise RuntimeError("Please import your get_latest_submission from sentiment.py into the PYTHONPATH.")

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

# ----------------------------------------------------------------------
# Helper utilities (sentiment, vwap, sma, atr, kelly, relevance)
# ----------------------------------------------------------------------

# Keep your original simple sentiment keywords
BULL_KEYWORDS = {"buy", "bull", "moon", "long", "pump", "call", "rocket", "tendies", "alpha", "hodl"}
BEAR_KEYWORDS = {"sell", "bear", "short", "dump", "exit", "weak", "stop", "down", "rekt", "fud"}

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

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (ATR)"""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def fractional_kelly(p: float, b: float, fraction: float = 0.5) -> float:
    """Return position fraction of bankroll using fractional Kelly."""
    q = 1 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return max(0.0, f_full) * fraction

def score_to_p(raw_score: float, alpha: float = 0.4, p_min: float = 0.05, p_max: float = 0.95) -> float:
    p = 0.5 + alpha * raw_score
    return max(p_min, min(p_max, p))

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
            pattern = r'\b(' + re.escape(ticker_low or "") + '|' + re.escape(company_low or "") + r')\b.*\b(price|buy|sell|shares|stock|short|long|earnings|dividend|guidance|ipo|merger)\b'
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

# ----------------------------------------------------------------------
# Crypto mapping for BTC and SOL
# ----------------------------------------------------------------------
CRYPTO_TICKER_MAP = {
    "BTC": {
        "pair": "BTC-USD",
        "name": "bitcoin",
        # include common words, cashtag, slang and the corn emoji 🌽 used for BTC
        "aliases": ["btc", "bitcoin", "$btc", "sats", "hodl", "corn", "🌽"],
    },
    "SOL": {
        "pair": "SOL-USD",
        "name": "solana",
        "aliases": ["sol", "solana", "$sol"],
    },
}

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Reddit→Market Recs", layout="wide")
st.title("Reddit-driven Stock & Crypto Recommendation (experimental)")
st.markdown(
    """
**Not financial advice.** This is an experimental tool that combines Reddit comment sentiment and simple technical rules to produce *suggestions* only. 
Always do your own research. (See: legal disclaimer.)
"""
)

# Sidebar inputs (kept from your original file)
st.sidebar.header("Inputs")
username = st.sidebar.text_input("Reddit user to observe", value="wsbapp")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", value="AAPL")
use_yf = st.sidebar.checkbox("Auto-fetch market data with yfinance", value=True)
current_price_input = st.sidebar.number_input("Current price (override)", value=0.0, format="%.4f")
account_size = st.sidebar.number_input("Account size (USD)", value=1000.0, step=100.0)
reward_risk = st.sidebar.number_input("Expected reward:risk (b)", value=1.5, step=0.1)
kelly_fraction = st.sidebar.slider("Kelly fraction", min_value=0.1, max_value=1.0, value=0.5)
sma_short_window = st.sidebar.number_input("SMA short window (days)", value=10, step=1)
sma_long_window = st.sidebar.number_input("SMA long window (days)", value=50, step=1)

# Tabs: Stock and Crypto
tabs = st.tabs(["Stock", "Crypto"])

# ---------------- STOCK TAB ----------------
with tabs[0]:
    st.header("Stock: sentiment → recommendation")

    run = st.button("Analyze stock (use sidebar inputs)")

    if run:
        st.info("Fetching Reddit thread and comments...")
        try:
            submission = get_latest_submission(username)
        except Exception as e:
            st.error(f"Unable to fetch Reddit submissions: {e}")
            st.stop()

        st.markdown(f"**Latest post:** {submission.title if hasattr(submission, 'title') else '—'}")

        # get top-level comments (sorted by 'best')
        submission.comment_sort = "best"
        top_level = get_up_to_n_comments(submission, max_comments=400, batch=40, pause_between_batches=1.0)
        comments_texts = [c.body.replace("\n", " ") for c in top_level]

        # classify bots (batch)
        st.info("Detecting bot comments (Gemini)...")
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
                # if Gemini returned a JSON list or string, be defensive
                label = str(lab).upper()
            if "BOT" not in label:
                human_comments.append(txt)

        st.write(f"Collected {len(comments_texts)} top-level comments — {len(human_comments)} classified as human.")

        # sentiment
        n_bull, n_bear, n_tot, score = simple_sentiment_score(human_comments)
        st.metric("Sentiment score (normalised)", f"{score:.3f}")
        st.write(f"Bullish: {n_bull}, Bearish: {n_bear}, Total human comments: {n_tot}")

        # Estimate p from sentiment
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

# ---------------- CRYPTO TAB ----------------
with tabs[1]:
    st.header("Crypto: BTC / SOL (intraday, ATR-based stops)")

    crypto_choice = st.selectbox("Choose crypto", options=["BTC", "SOL", "Custom"], index=0)
    if crypto_choice == "Custom":
        crypto_ticker = st.text_input("Custom pair (yfinance, e.g. ETH-USD)", value="BTC-USD")
        crypto_name = st.text_input("Asset name for relevance (e.g. ethereum)", value="crypto")
        aliases_text = st.text_input("Aliases (comma-separated)", value="")
        aliases = [a.strip() for a in aliases_text.split(",") if a.strip()]
    else:
        info = CRYPTO_TICKER_MAP[crypto_choice]
        crypto_ticker = info["pair"]
        crypto_name = info["name"]
        aliases = info["aliases"]

    interval = st.selectbox("OHLC interval", options=["1h", "4h"], index=0)
    period_days = st.number_input("Period days to fetch", value=14, step=1)
    sma_short_h = st.number_input("SMA short (bars)", value=24, step=1)
    sma_long_h = st.number_input("SMA long (bars)", value=72, step=1)
    atr_mult = st.number_input("ATR multiplier for stop", value=1.5, step=0.1)
    kelly_frac_crypto = st.slider("Kelly fraction (crypto)", 0.01, 0.5, 0.1)
    account_size_crypto = st.number_input("Account size (USD) for crypto", value=1000.0, step=100.0)
    run_crypto = st.button("Analyze crypto")

    if run_crypto:
        st.info("Fetching Reddit thread and comments...")
        try:
            submission = get_latest_submission(username)
        except Exception as e:
            st.error(f"Unable to fetch Reddit submissions: {e}")
            st.stop()

        # fetch comments (allow more for crypto)
        try:
            comments_objs = get_up_to_n_comments(submission, max_comments=400, batch=60, pause_between_batches=1.0)
            comments_texts = [c.body.replace("\n", " ") for c in comments_objs]
        except Exception:
            try:
                submission.comments.replace_more(limit=0)
                comments_texts = [c.body.replace("\n", " ") for c in submission.comments.list()]
            except Exception as e:
                st.error("Failed to load comments: " + str(e))
                comments_texts = []

        st.write(f"Collected {len(comments_texts)} comments (raw)")

        # relevance filter for crypto: use crypto_choice ticker token (or custom ticker's left side)
        relevance_ticker = crypto_choice if crypto_choice != "Custom" else crypto_ticker.split("-")[0]
        relevant = filter_comments_by_relevance(comments_texts, relevance_ticker, crypto_name, aliases=aliases)
        st.write(f"Relevant comments: {len(relevant)} — running bot detection...")

        # classify bots
        try:
            classifications = classify_comments_with_gemini(relevant)
        except Exception as e:
            st.warning("Gemini classification failed — proceeding without bot filtering.")
            classifications = [{"index": i + 1, "label": "HUMAN"} for i in range(len(relevant))]

        human_comments = []
        for i, txt in enumerate(relevant):
            lab = classifications[i] if i < len(classifications) else {"label": "HUMAN"}
            label = lab.get("label", "HUMAN") if isinstance(lab, dict) else str(lab).upper()
            if "BOT" not in label:
                human_comments.append(txt)

        st.write(f"{len(human_comments)} comments classified as human & relevant.")

        # sentiment
        n_bull, n_bear, n_tot, raw_score = simple_sentiment_score(human_comments)
        st.metric("Sentiment score (normalised)", f"{raw_score:.3f}")
        st.write(f"Bullish: {n_bull}, Bearish: {n_bear}, Total human relevant: {n_tot}")

        # map to p (more conservative alpha for crypto)
        p_est = score_to_p(raw_score, alpha=0.3)
        st.write(f"Estimated win probability p = {p_est:.2f}")

        # fetch intraday market data via yfinance
        st.info(f"Fetching market data for {crypto_ticker} (interval {interval})...")
        try:
            df = yf.Ticker(crypto_ticker).history(period=f"{period_days}d", interval=interval)
            if df.empty:
                st.warning("No market data returned.")
                df = None
                current_price = 0.0
            else:
                df = df.rename(columns={c: c.capitalize() for c in df.columns})
                df = df[["High", "Low", "Close", "Volume"]]
                df["VWAP"] = compute_vwap(df)
                df["SMA_short"] = sma(df["Close"], sma_short_h)
                df["SMA_long"] = sma(df["Close"], sma_long_h)
                df["ATR"] = atr(df, window=14)
                current_price = df["Close"].iloc[-1]
                st.write(f"Using price = {current_price:.4f}")
        except Exception as e:
            st.warning("Market data fetch failed: " + str(e))
            df = None
            current_price = 0.0

        # sizing & ATR stop
        b = st.number_input("Reward:risk (b)", value=1.5, step=0.1, key="crypto_b")
        f = fractional_kelly(p_est, b, fraction=kelly_frac_crypto)
        pos_usd = f * account_size_crypto
        st.subheader("Suggested sizing (Kelly-based, crypto)")
        st.write(f"Fraction of bankroll (fractional Kelly) = {f:.4f}")
        st.write(f"Suggested position size = ${pos_usd:,.2f} (account ${account_size_crypto:,.0f})")

        # ATR stop / TP
        if df is not None and current_price > 0:
            last_atr = df["ATR"].iloc[-1]
            if np.isnan(last_atr) or last_atr == 0:
                st.warning("ATR not available — using fixed percent stop")
                r = st.number_input("Stop-loss fraction r (fallback)", value=0.05, step=0.01, key="crypto_r")
                stop_loss = current_price * (1 - r)
                take_profit = current_price * (1 + r * b)
            else:
                stop_loss = current_price - atr_mult * last_atr
                stop_loss = max(0.0001, stop_loss)
                take_profit = current_price + (current_price - stop_loss) * b

            st.subheader("Recommendation (crypto)")
            st.write(f"Entry: {current_price:.4f}")
            st.write(f"ATR(14) = {last_atr:.4f}")
            st.write(f"Stop-loss (ATR x {atr_mult}): {stop_loss:.4f}")
            st.write(f"Take-profit: {take_profit:.4f}")
        else:
            st.info("No market data available to compute ATR-based stops.")

st.markdown("---")
st.caption("Experimental tool — not financial advice. Use with caution.")
