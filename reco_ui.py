# reco_ui.py
import os
import re
import math
import time
from typing import List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import praw

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

# Remove 'bull' token from BULL_KEYWORDS to avoid ticker collision
BULL_KEYWORDS = {"buy", "moon", "long", "pump", "call", "rocket", "tendies", "alpha", "hodl", "bullish"}
BEAR_KEYWORDS = {"sell", "bear", "short", "dump", "exit", "weak", "stop", "down", "rekt", "fud"}

def remove_ignore_tokens(text: str, ignore_tokens: List[str] | None):
    t = (text or "")
    t = t.replace("\u200b", "")  # strip zero-width if present
    t_low = t.lower()
    if not ignore_tokens:
        return t_low
    for token in ignore_tokens:
        if not token:
            continue
        tok = token.lower().strip()
        t_low = re.sub(r'(\$)?\b' + re.escape(tok) + r'\b', ' ', t_low)
    t_low = re.sub(r'\s+', ' ', t_low).strip()
    return t_low

def simple_sentiment_score(comments: List[str], ignore_tokens: List[str] | None = None):
    """
    Return (n_bull, n_bear, n_total, score_norm) for list of comment strings.
    ignore_tokens: list of tokens (tickers, company names, aliases) to strip before checking sentiment keywords.
    """
    n_bull = n_bear = 0
    for c in comments:
        cleaned = remove_ignore_tokens(c, ignore_tokens)
        for k in BULL_KEYWORDS:
            if k in cleaned:
                n_bull += 1
                break
        for k in BEAR_KEYWORDS:
            if k in cleaned:
                n_bear += 1
                break
    n_total = len(comments)
    score = (n_bull - n_bear) / max(1, n_total)
    return n_bull, n_bear, n_total, score

def compute_vwap(df: pd.DataFrame):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    return pv.cumsum() / df["Volume"].cumsum()

def sma(series: pd.Series, window: int):
    return series.rolling(window).mean()

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def fractional_kelly(p: float, b: float, fraction: float = 0.5) -> float:
    q = 1 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    return max(0.0, f_full) * fraction

def score_to_p(raw_score: float, alpha: float = 0.4, p_min: float = 0.05, p_max: float = 0.95) -> float:
    p = 0.5 + alpha * raw_score
    return max(p_min, min(p_max, p))

def filter_comments_by_relevance(comments: List[str], ticker: str, company: str, aliases: List[str] = None) -> List[str]:
    aliases = aliases or []
    ticker_low = (ticker or "").lower()
    company_low = (company or "").lower()

    def is_relevant(text: str) -> bool:
        t = (text or "").lower()
        if ticker_low and re.search(r'\$' + re.escape(ticker_low), t):
            return True
        if ticker_low and re.search(r'\b' + re.escape(ticker_low) + r'\b', t):
            return True
        if company_low and company_low in t:
            return True
        for a in aliases:
            if a and a.lower() in t:
                return True
        if ticker_low or company_low:
            pattern = r'\b(' + re.escape(ticker_low or "") + '|' + re.escape(company_low or "") + r')\b.*\b(price|buy|sell|shares|stock|short|long|earnings|dividend|guidance|ipo|merger|hodl|sats|pump|dump|fud|moon)\b'
            if re.search(pattern, t):
                return True
        return False

    return [c for c in comments if is_relevant(c)]

def get_up_to_n_comments(submission, max_comments=500, batch=25, pause_between_batches=0.5):
    safe_replace_more(submission.comments, limit=0)
    comments = submission.comments.list()
    if len(comments) >= max_comments:
        return comments[:max_comments]
    while len(comments) < max_comments:
        prev_len = len(comments)
        try:
            safe_replace_more(submission.comments, limit=batch)
        except Exception as e:
            st.write("Error expanding more comments:", e)
            break
        if pause_between_batches:
            time.sleep(pause_between_batches)
        comments = submission.comments.list()
        if len(comments) == prev_len:
            break
    return comments[:max_comments]

def debug_relevance_examples(all_comments: List[str], relevant_comments: List[str], n: int = 6) -> None:
    """
    Print a few examples of relevant and non-relevant comments in Streamlit
    so you can tune your relevance filters.
    """
    st.write("**Sample relevant (up to n):**")
    if not relevant_comments:
        st.write("_(none)_")
    else:
        for c in relevant_comments[:n]:
            # truncate to keep the UI compact
            st.write("-", (c or "")[:300])

    st.write("**Sample non-relevant (up to n):**")
    # avoid O(n^2) in pathological cases: build a set for fast membership
    rel_set = set(relevant_comments)
    nonrel = [c for c in all_comments if c not in rel_set]
    if not nonrel:
        st.write("_(none)_")
    else:
        for c in nonrel[:n]:
            st.write("-", (c or "")[:300])

# ---------------------------------------
# CRYPTO ALIASES (include corn 🌽 for BTC)
# ---------------------------------------
CRYPTO_TICKER_MAP = {
    "BTC": {"pair": "BTC-USD", "name": "bitcoin", "aliases": ["btc", "bitcoin", "$btc", "sats", "hodl", "corn", "🌽"]},
    "SOL": {"pair": "SOL-USD", "name": "solana", "aliases": ["sol", "solana", "$sol"]},
}

# ---------------------------------------
# Top-ticker discovery helpers
# ---------------------------------------
CASHTAG_RE = re.compile(r'\$([A-Z]{1,6})\b')
UPPER_TICKER_RE = re.compile(r'\b([A-Z]{2,6})\b')
FINANCE_WORDS = r'(price|buy|sell|shares|stock|short|long|earnings|dividend|ipo|pump|dump|hodl|sats|moon|chart)'

def extract_tickers_from_text(text: str) -> List[str]:
    if not text:
        return []
    found = set()
    # cashtags
    for m in CASHTAG_RE.findall(text):
        found.add(m.upper())
    # uppercase tokens (cautious)
    for m in UPPER_TICKER_RE.findall(text):
        token = m.upper()
        # skip very common words
        if token.lower() in {"the","and","for","you","not","with","this","that","are","from","about","but","can","has","have","will","all","our","how","who","what","when","where","why","its","in","your","new","use","was","is","it"}:
            continue
        if len(token) <= 3:
            pattern = re.compile(r'(' + re.escape(token) + r').{0,40}\b' + FINANCE_WORDS + r'\b', re.IGNORECASE)
            if not pattern.search(text):
                pattern2 = re.compile(r'\b' + FINANCE_WORDS + r'\b.{0,40}(' + re.escape(token) + r')', re.IGNORECASE)
                if not pattern2.search(text):
                    continue
        found.add(token)
    return list(found)

def find_top_tickers_from_subreddits(reddit_client, subreddits: List[str], hours: int = 24, max_sub_per_sub: int = 150, max_comments_per_submission: int = 200, include_titles: bool = True, validate_with_yf: bool = False, validate_limit: int = 5) -> List[Tuple[str,int]]:
    now = time.time()
    cutoff = now - hours * 3600
    counter = {}
    def update_counter(ticks):
        for t in ticks:
            counter[t] = counter.get(t, 0) + 1

    for sub in subreddits:
        try:
            subreddit = reddit_client.subreddit(sub)
        except Exception:
            continue
        try:
            for submission in subreddit.new(limit=max_sub_per_sub):
                created = getattr(submission, "created_utc", None)
                if created is None or created < cutoff:
                    continue
                if include_titles:
                    tx = (getattr(submission, "title", "") or "") + "\n" + (getattr(submission, "selftext", "") or "")
                    ticks = extract_tickers_from_text(tx)
                    update_counter(ticks)
                # comments
                try:
                    # try to use your helper if present
                    comment_objs = []
                    if 'get_up_to_n_comments' in globals() and callable(get_up_to_n_comments):
                        comment_objs = get_up_to_n_comments(submission, max_comments=max_comments_per_submission, batch=25, pause_between_batches=0.3)
                        comment_texts = [c.body for c in comment_objs if getattr(c, "body", None)]
                    else:
                        submission.comments.replace_more(limit=0)
                        comment_texts = [c.body for c in submission.comments.list()[:max_comments_per_submission] if getattr(c, "body", None)]
                except Exception:
                    comment_texts = []
                for c in comment_texts:
                    ticks = extract_tickers_from_text(c)
                    update_counter(ticks)
        except Exception:
            # continue to next subreddit on errors (rate limits etc.)
            continue

    # sort
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # optional validate with yfinance
    if validate_with_yf and ranked:
        validated = []
        for t, count in ranked:
            if len(validated) >= validate_limit:
                break
            ok = False
            try:
                df = yf.Ticker(t).history(period="1d", interval="1d")
                if df is None or df.empty:
                    df = yf.download(t + "-USD", period="1d", interval="1d", progress=False)
                if df is not None and not df.empty:
                    ok = True
            except Exception:
                ok = False
            if ok:
                validated.append((t, count))
        if validated:
            return validated
    return ranked

# ---------------------------------------
# Streamlit UI (keeps your original layout and adds Top tickers tab)
# ---------------------------------------
st.set_page_config(page_title="Reddit→Market Recs", layout="wide")
st.title("Reddit-driven Stock & Crypto Recommendation (experimental)")
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
company_name = st.sidebar.text_input("Company name (for relevance, optional)", value="")
use_yf = st.sidebar.checkbox("Auto-fetch market data with yfinance", value=True)
current_price_input = st.sidebar.number_input("Current price (override)", value=0.0, format="%.4f")
account_size = st.sidebar.number_input("Account size (USD)", value=1000.0, step=100.0)
reward_risk = st.sidebar.number_input("Expected reward:risk (b)", value=1.5, step=0.1)
kelly_fraction = st.sidebar.slider("Kelly fraction", min_value=0.1, max_value=1.0, value=0.5)
sma_short_window = st.sidebar.number_input("SMA short window (days)", value=10, step=1)
sma_long_window = st.sidebar.number_input("SMA long window (days)", value=50, step=1)
filter_before_classify = st.sidebar.checkbox("Filter by ticker/company before classification (recommended)", value=True)
show_relevance_debug = st.sidebar.checkbox("Show relevance debug examples", value=False)

# Tabs: Stock, Crypto, Top tickers
tabs = st.tabs(["Stock", "Crypto", "Top tickers"])

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
        submission.comment_sort = "best"
        top_level = get_up_to_n_comments(submission, max_comments=500, batch=40, pause_between_batches=1.0)
        comments_texts = [c.body.replace("\n", " ") for c in top_level]
        st.write(f"Collected {len(comments_texts)} top-level comments (raw)")

        if filter_before_classify:
            company_for_relevance = company_name if company_name else ticker
            aliases = [ticker.lower()]
            if company_for_relevance and company_for_relevance.lower() != ticker.lower():
                aliases.append(company_for_relevance.lower())
            if ticker.lower() == "sol":
                aliases.extend(["sol", "solana"])
            if ticker.lower() == "btc":
                aliases.extend(["btc", "bitcoin", "corn", "🌽", "hodl", "sats"])

            relevant = filter_comments_by_relevance(comments_texts, ticker, company_for_relevance, aliases=aliases)
            st.write(f"Relevant comments by ticker/company: {len(relevant)} out of {len(comments_texts)}")

            if show_relevance_debug:
                debug_relevance_examples(comments_texts, relevant, n=6)

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
                    label = str(lab).upper()
                if "BOT" not in label:
                    human_comments.append(txt)

            st.write(f"{len(human_comments)} comments classified as human & relevant.")
        else:
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

        # sentiment (ignore tokens so tickers don't pollute counts)
        company_for_relevance = company_name if company_name else ticker
        ignore_tokens = [ticker.lower()]
        if company_for_relevance and company_for_relevance.lower() != ticker.lower():
            ignore_tokens.append(company_for_relevance.lower())
        n_bull, n_bear, n_tot, score = simple_sentiment_score(human_comments, ignore_tokens)
        st.metric("Sentiment score (normalised)", f"{score:.3f}")
        st.write(f"Bullish: {n_bull}, Bearish: {n_bear}, Total human comments: {n_tot}")

        # p estimation and market data (robust fetch)
        p_est = 0.5 + 0.4 * score
        p_est = max(0.05, min(0.95, p_est))
        hist = None
        current_price = 0.0
        if use_yf:
            st.info("Fetching market data from Yahoo Finance...")
            st.write("Ticker requested:", ticker)
            try:
                df = yf.Ticker(ticker).history(period="90d", interval="1d")
                if df is None or df.empty:
                    st.write("Primary history() returned empty — trying yf.download fallback...")
                    df = yf.download(ticker, period="90d", interval="1d", progress=False)

                if df is None or df.empty:
                    st.warning("No market data returned for ticker. Check ticker spelling and availability on Yahoo Finance.")
                    current_price = current_price_input if current_price_input > 0 else 0.0
                else:
                    df = df.rename(columns={c: c.capitalize() for c in df.columns})
                    for col in ["High", "Low", "Close", "Volume"]:
                        if col not in df.columns:
                            st.warning(f"Market data missing column: {col}. Columns present: {list(df.columns)}")
                    df = df[["High", "Low", "Close", "Volume"]]
                    df["VWAP"] = compute_vwap(df)
                    df["SMA_short"] = sma(df["Close"], sma_short_window)
                    df["SMA_long"] = sma(df["Close"], sma_long_window)
                    current_price = current_price_input if current_price_input > 0 else float(df["Close"].iloc[-1])
                    hist = df
                if hist is not None:
                    st.write("Market data (last 3 rows):")
                    st.dataframe(hist.tail(3))
            except Exception as e:
                st.warning("yfinance fetch failed: " + str(e))
                st.write("Exception details (for debugging):", repr(e))
                current_price = current_price_input if current_price_input > 0 else 0.0
        else:
            current_price = current_price_input if current_price_input > 0 else 0.0

        st.write(f"Using price = {current_price:.4f}")

        pos_frac = fractional_kelly(p=p_est, b=reward_risk, fraction=kelly_fraction)
        pos_usd = pos_frac * account_size
        st.subheader("Suggested position sizing (Kelly-based)")
        st.write(f"Estimated win probability p = {p_est:.2f}")
        st.write(f"Reward:risk b = {reward_risk:.2f}")
        st.write(f"Fraction of bankroll (fractional Kelly) = {pos_frac:.4f}")
        st.write(f"Suggested position size = ${pos_usd:,.2f} (on account ${account_size:,.0f})")

        # Entry/exit logic
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

            if score > 0.15 and current_price > sma_short:
                decision = "BUY (momentum)"
            elif score > 0.2 and current_price < vwap:
                decision = "BUY (dip)"
            elif score < -0.2 and current_price < sma_short:
                decision = "SELL (weak)"

            r = 0.03
            entry_price = current_price
            stop_loss = entry_price * (1 - r)
            take_profit = entry_price * (1 + r * reward_risk)
        else:
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

        try:
            comments_objs = get_up_to_n_comments(submission, max_comments=800, batch=60, pause_between_batches=1.0)
            comments_texts = [c.body.replace("\n", " ") for c in comments_objs]
        except Exception:
            try:
                submission.comments.replace_more(limit=0)
                comments_texts = [c.body.replace("\n", " ") for c in submission.comments.list()]
            except Exception as e:
                st.error("Failed to load comments: " + str(e))
                comments_texts = []

        st.write(f"Collected {len(comments_texts)} comments (raw)")

        relevance_ticker = crypto_choice if crypto_choice != "Custom" else crypto_ticker.split("-")[0]
        relevant = filter_comments_by_relevance(comments_texts, relevance_ticker, crypto_name, aliases=aliases)
        st.write(f"Relevant comments: {len(relevant)} — running bot detection...")

        try:
            classifications = classify_comments_with_gemini(relevant)
        except Exception as e:
            st.warning("Gemini classification failed — proceeding without bot filtering.")
            classifications = [{"index": i+1, "label": "HUMAN"} for i in range(len(relevant))]

        human_comments = []
        for i, txt in enumerate(relevant):
            lab = classifications[i] if i < len(classifications) else {"label": "HUMAN"}
            label = lab.get("label", "HUMAN") if isinstance(lab, dict) else str(lab).upper()
            if "BOT" not in label:
                human_comments.append(txt)

        st.write(f"{len(human_comments)} comments classified as human & relevant.")

        crypto_ignore = [relevance_ticker.lower()] + [a.lower() for a in aliases if a]
        n_bull, n_bear, n_tot, raw_score = simple_sentiment_score(human_comments, ignore_tokens=crypto_ignore)
        st.metric("Sentiment score (normalised)", f"{raw_score:.3f}")
        st.write(f"Bullish: {n_bull}, Bearish: {n_bear}, Total human relevant: {n_tot}")

        p_est = score_to_p(raw_score, alpha=0.3)
        st.write(f"Estimated win probability p = {p_est:.2f}")

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

        b = st.number_input("Reward:risk (b)", value=1.5, step=0.1, key="crypto_b")
        f = fractional_kelly(p_est, b, fraction=kelly_frac_crypto)
        pos_usd = f * account_size_crypto
        st.subheader("Suggested sizing (Kelly-based, crypto)")
        st.write(f"Fraction of bankroll (fractional Kelly) = {f:.4f}")
        st.write(f"Suggested position size = ${pos_usd:,.2f} (account ${account_size_crypto:,.0f})")

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

# ---------------- TOP TICKERS TAB ----------------
with tabs[2]:
    st.header("Top tickers (today) — quick discovery")
    st.markdown(
        "Scan recent posts & comments across a set of subreddits to find the most-mentioned tickers "
        "today. This uses cashtags and uppercase token heuristics; results can be validated via yfinance."
    )

    # scanner inputs
    subreddits_input = st.text_input("Subreddits (comma-separated)", value="wallstreetbets, stocks, investing, CryptoCurrency")
    hours = st.slider("Hours back", min_value=1, max_value=72, value=24, step=1)
    max_sub_per_sub = st.number_input("Max submissions per subreddit", value=120, step=10)
    max_comments_per_submission = st.number_input("Max comments per submission", value=200, step=50)
    top_n = st.number_input("Top N tickers to show", value=25, step=5)
    validate_yf = st.checkbox("Validate top candidates with yfinance (slower)", value=False)
    scan_btn = st.button("Scan now for top tickers")

    if scan_btn:
        subs = [s.strip() for s in subreddits_input.split(",") if s.strip()]
        st.info(f"Scanning subreddits: {subs} for the last {hours} hours — this may take a little while.")
        # create a lightweight PRAW client (read-only works without refresh token in many cases)
        try:
            praw_kwargs = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "user_agent": USER_AGENT}
            if REFRESH_TOKEN:
                praw_kwargs["refresh_token"] = REFRESH_TOKEN
            reddit_client = praw.Reddit(**praw_kwargs)
        except Exception as e:
            st.error(f"Failed to create Reddit client: {e}")
            st.stop()

        # run the scanner
        with st.spinner("Scanning..."):
            try:
                ranked = find_top_tickers_from_subreddits(
                    reddit_client,
                    subreddits=subs,
                    hours=hours,
                    max_sub_per_sub=max_sub_per_sub,
                    max_comments_per_submission=max_comments_per_submission,
                    include_titles=True,
                    validate_with_yf=validate_yf,
                    validate_limit=10
                )
            except Exception as e:
                st.error(f"Ticker discovery failed: {e}")
                ranked = []

        if not ranked:
            st.warning("No tickers found by the scanner. Try increasing max_submissions or changing subreddits.")
        else:
            df_rank = pd.DataFrame(ranked, columns=["Ticker", "Mentions"])
            st.subheader("Top tickers")
            st.dataframe(df_rank.head(top_n))

            # allow selecting tickers to quick-analyze
            selected = st.multiselect("Select tickers to Quick Analyze (sentiment + simple sizing)", options=df_rank["Ticker"].tolist(), default=df_rank["Ticker"].tolist()[:3])
            if selected:
                analyze_btn = st.button("Quick analyze selected tickers")
                if analyze_btn:
                    # for each selected ticker run a lightweight analyzer (search recent posts for cashtag/ticker)
                    results = []
                    for t in selected:
                        st.write(f"--- Analyzing {t} ---")
                        # collect comments mentioning ticker (simple subreddit search)
                        combined_comments = []
                        for sub in subs:
                            try:
                                sr = reddit_client.subreddit(sub)
                                query = f'"${t}" OR "{t}"'
                                for post in sr.search(query, sort='new', time_filter='day', limit=10):
                                    try:
                                        post.comments.replace_more(limit=0)
                                        comment_bodies = [c.body for c in post.comments.list()[:100] if getattr(c, "body", None)]
                                    except Exception:
                                        comment_bodies = []
                                    combined_comments.extend(comment_bodies)
                            except Exception:
                                continue
                        combined_comments = list(dict.fromkeys(combined_comments))  # dedupe
                        st.write(f"Collected {len(combined_comments)} comments for {t}")
                        # relevance: keep comments that include cashtag or ticker token
                        relevant = [c for c in combined_comments if (f'${t.lower()}' in c.lower()) or re.search(r'\b' + re.escape(t.lower()) + r'\b', c.lower())]
                        st.write(f"Relevant comments: {len(relevant)}")
                        # bot classification (best effort)
                        try:
                            cls = classify_comments_with_gemini(relevant)
                        except Exception:
                            cls = [{"index": i+1, "label":"HUMAN"} for i in range(len(relevant))]
                        human_comments = []
                        for i, txt in enumerate(relevant):
                            lab = cls[i] if i < len(cls) else {"label":"HUMAN"}
                            label = lab.get("label","HUMAN") if isinstance(lab, dict) else str(lab)
                            if "BOT" not in label:
                                human_comments.append(txt)
                        st.write(f"Human classified comments: {len(human_comments)}")
                        ignore_tokens = [t.lower()]
                        n_bull, n_bear, n_tot, raw_score = simple_sentiment_score(human_comments, ignore_tokens)
                        st.write(f"bulls={n_bull}, bears={n_bear}, total={n_tot}, score={raw_score:.3f}")
                        p_est_local = score_to_p(raw_score)
                        pos_frac_local = fractional_kelly(p_est_local, b=reward_risk, fraction=kelly_fraction)
                        pos_usd_local = pos_frac_local * account_size
                        st.write(f"Estimated p = {p_est_local:.2f}; suggested position = ${pos_usd_local:,.2f}")
                        results.append((t, n_bull, n_bear, n_tot, raw_score, p_est_local, pos_usd_local))
                    # show summary table
                    summ = pd.DataFrame(results, columns=["Ticker","Bull","Bear","Total","Score","P_est","Suggested_$"])
                    st.dataframe(summ)

st.markdown("---")
st.caption("Experimental tool — not financial advice. Use with caution.")
