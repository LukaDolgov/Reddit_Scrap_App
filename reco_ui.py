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
SLANG_BLACKLIST = {
    "YOLO","FOMO","LMAO","LOL","WTF","OMG","TBT","FYI","TBH","IMO","IDK","BULL", "MOON", "LEAPS", "CASINO", "CHIPS", "OP"
}
COMMON_WORDS_LOW = {"the","and","for","you","not","with","this","that","are","from","about","but","can","has","have","will","all","our","how","who","what","when","where","why","its","in","your","new","use","was","is","it"}
# keep lowercase set for quick checks
SLANG_BLACKLIST_LOW = {s.lower() for s in SLANG_BLACKLIST}
def extract_tickers_from_text(text: str) -> List[str]:
    """
    Extract candidate tickers from text:
      - Always include $CASHTAG tokens.
      - Include uppercase tokens (2-6 letters) only if:
         * token not in slang blacklist AND
         * token not used mostly as lowercase in the same text (heuristic), AND
         * (token is in finance context nearby OR a later YF validation will confirm it).
    Returns uppercase tickers (no $).
    """
    if not text:
        return []

    found = set()
    text_str = text or ""
    lower_text = text_str.lower()

    # 1) cashtags (high confidence)
    for m in CASHTAG_RE.findall(text_str):
        found.add(m.upper())

    # 2) uppercase tokens (cautious)
    # Find all uppercase token candidates
    for m in UPPER_TICKER_RE.findall(text_str):
        token = m.upper()
        token_low = token.lower()

        # quick slang / common-word rejection
        if token_low in SLANG_BLACKLIST_LOW:
            continue

        # avoid common English words (existing check)
        if token_low in COMMON_WORDS_LOW:
            continue

        # heuristic: if token appears in lowercase (e.g., "yolo" in text) many times -> it's not a ticker
        # Count occurrences of token both cases
        # if lowercase occurrences > 0 and is same or more than uppercase occurrences -> skip
        ups = len(re.findall(r'\b' + re.escape(token) + r'\b', text_str))
        lows = len(re.findall(r'\b' + re.escape(token_low) + r'\b', lower_text))
        # if token mostly appears lowercase (natural-language usage), skip
        if lows > ups and lows >= 2:
            continue

        # short tokens need finance context nearby (2-3 characters)
        if len(token) <= 3:
            pattern = re.compile(r'(' + re.escape(token) + r').{0,40}\b' + FINANCE_WORDS + r'\b', re.IGNORECASE)
            if not pattern.search(text_str):
                pattern2 = re.compile(r'\b' + FINANCE_WORDS + r'\b.{0,40}(' + re.escape(token) + r')', re.IGNORECASE)
                if not pattern2.search(text_str):
                    # don't include now — it might be valid across corpus and could be validated later by yfinance
                    continue

        # Passed checks — accept token
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
    st.header("Top tickers (today) — quick discovery (with confidence & weighting)")
    st.markdown(
        "Scan recent posts & comments across a set of subreddits to find the most-mentioned tickers "
        "today. This tab now shows intermediate counts, a Wilson CI for bullish fraction, and upvote-weighted scoring."
    )

    # ---------- scanner inputs ----------
    subreddits_input = st.text_input("Subreddits (comma-separated)", value="wallstreetbets, stocks, investing, CryptoCurrency")
    hours = st.slider("Hours back", min_value=1, max_value=72, value=24, step=1)
    max_sub_per_sub = st.number_input("Max submissions per subreddit", value=120, step=10)
    max_comments_per_submission = st.number_input("Max comments per submission", value=200, step=50)
    top_n = st.number_input("Top N tickers to show", value=25, step=5)
    validate_yf = st.checkbox("Validate top candidates with yfinance (slower)", value=False)
    scan_btn = st.button("Scan now for top tickers")

    # ---------- tuning for sentiment scan & confidence ----------
    posts_per_sub_for_scan = st.number_input("Posts/sample per subreddit (per ticker)", value=6, step=1)
    comments_per_post_for_scan = st.number_input("Comments/sample per post", value=80, step=10)
    min_n_total = st.number_input("Minimum n_total for high-confidence", value=30, step=1)
    use_weighting = st.checkbox("Use upvote-weighting in additional metrics", value=True)

    # ---------- utility helpers ----------
    def wilson_interval(k: int, n: int, z: float = 1.96):
        """Wilson score interval for proportion k/n (returns lo, hi)."""
        if n == 0:
            return 0.0, 1.0
        phat = k / n
        denom = 1 + z*z/n
        centre = phat + z*z/(2*n)
        adj = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n)
        lo = (centre - adj) / denom
        hi = (centre + adj) / denom
        return max(0.0, lo), min(1.0, hi)

    def comment_weight(score: int) -> float:
        """Transform comment.score (upvotes) -> positive weight. Use log1p to damp extreme votes."""
        try:
            s = max(0, int(score))
            return 1.0 + math.log1p(s)
        except Exception:
            return 1.0

    def classify_comment_token_label(text: str, ignore_tokens: List[str] | None = None) -> str:
        """
        Very small deterministic rule: return 'BULL', 'BEAR', or 'NEUTRAL' for a single comment.
        Uses the same keyword lists as simple_sentiment_score but runs per-comment.
        """
        cleaned = remove_ignore_tokens(text, ignore_tokens)
        cleaned = (cleaned or "").lower()
        for k in BULL_KEYWORDS:
            if k in cleaned:
                return "BULL"
        for k in BEAR_KEYWORDS:
            if k in cleaned:
                return "BEAR"
        return "NEUTRAL"

    # ---------- run scan ----------
    if scan_btn:
        subs = [s.strip() for s in subreddits_input.split(",") if s.strip()]
        st.info(f"Scanning subreddits: {subs} for the last {hours} hours — this may take a little while.")
        # create reddit client (reuse credentials)
        try:
            praw_kwargs = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, "user_agent": USER_AGENT}
            if REFRESH_TOKEN:
                praw_kwargs["refresh_token"] = REFRESH_TOKEN
            reddit_client = praw.Reddit(**praw_kwargs)
        except Exception as e:
            st.error(f"Failed to create Reddit client: {e}")
            st.stop()

        # run discovery (uses your find_top_tickers_from_subreddits helper from above)
        with st.spinner("Scanning for tickers..."):
            try:
                ranked = find_top_tickers_from_subreddits(
                    reddit_client,
                    subreddits=subs,
                    hours=hours,
                    max_sub_per_sub=max_sub_per_sub,
                    max_comments_per_submission=max_comments_per_submission,
                    include_titles=True,
                    validate_with_yf=validate_yf,
                    validate_limit=25
                )
            except Exception as e:
                st.error(f"Ticker discovery failed: {e}")
                ranked = []

        if not ranked:
            st.warning("No tickers found by the scanner. Try increasing max_submissions or changing subreddits.")
            st.stop()

        # Normalize ranked into DataFrame
        df_rank = pd.DataFrame(ranked, columns=["Ticker", "Mentions"])
        df_rank = df_rank.sort_values("Mentions", ascending=False).reset_index(drop=True)
        st.subheader("Top tickers (raw)")
        st.dataframe(df_rank.head(int(top_n)))

        # ---------- per-ticker sentiment scanner (improved: keep score + dedupe by text keeping max upvote) ----------
        def compute_sentiment_for_ticker_enhanced(ticker_symbol: str, reddit_client, subs_list: List[str], posts_per_sub=3, comments_per_post=50):
            ticker_lower = ticker_symbol.lower()
            # Collect comments as dict body->max_score to dedupe while keeping highest upvote
            collected = {}
            for sub in subs_list:
                try:
                    sr = reddit_client.subreddit(sub)
                    query = f'"${ticker_symbol}" OR "{ticker_symbol}"'
                    for post in sr.search(query, sort='new', time_filter='day', limit=posts_per_sub):
                        try:
                            post.comments.replace_more(limit=0)
                            # grab comment objects slice
                            comment_objs = post.comments.list()[:comments_per_post]
                            for c in comment_objs:
                                if not getattr(c, "body", None):
                                    continue
                                body = c.body.strip()
                                # update with max score for dedupe
                                score_val = getattr(c, "score", 0) or 0
                                prev = collected.get(body)
                                if prev is None or score_val > prev:
                                    collected[body] = score_val
                        except Exception:
                            continue
                except Exception:
                    continue

            # Build list of comment dicts preserving highest score per unique text
            combined_comments = [{"body": b, "score": s} for b, s in collected.items()]
            collected_mentions = len(combined_comments)

            # If nothing collected, return neutral row
            if collected_mentions == 0:
                return {
                    "ticker": ticker_symbol,
                    "collected_mentions": 0,
                    "relevant_count": 0,
                    "human_count": 0,
                    "n_bull": 0,
                    "n_bear": 0,
                    "n_total": 0,
                    "score": 0.0,
                    "weighted_score": 0.0,
                    "p_est": 0.5,
                    "wilson_lo": 0.0,
                    "wilson_hi": 1.0,
                    "suggested_usd": 0.0,
                    "low_confidence": True
                }

            # Relevance filter: keep comments mentioning cashtag OR whole-word ticker
            relevant_comments = []
            for item in combined_comments:
                low = (item["body"] or "").lower()
                if f'${ticker_lower}' in low or re.search(r'\b' + re.escape(ticker_lower) + r'\b', low):
                    relevant_comments.append(item)
            # fallback: use all if none matched relevance
            if not relevant_comments:
                relevant_comments = combined_comments[:]

            relevant_count = len(relevant_comments)

            # Bot classification (Gemini) - needs list of texts
            try:
                texts_for_classify = [it["body"] for it in relevant_comments]
                classif = classify_comments_with_gemini(texts_for_classify)
            except Exception:
                classif = [{"index": i + 1, "label": "HUMAN"} for i in range(len(relevant_comments))]

            # Build human_comments list of dicts (body, score)
            human_comments = []
            for i, it in enumerate(relevant_comments):
                lab = classif[i] if i < len(classif) else {"label": "HUMAN"}
                label = lab.get("label", "HUMAN") if isinstance(lab, dict) else str(lab)
                if "BOT" not in label:
                    human_comments.append(it)

            human_count = len(human_comments)

            # If no human comments, return low-confidence neutral
            if human_count == 0:
                return {
                    "ticker": ticker_symbol,
                    "collected_mentions": collected_mentions,
                    "relevant_count": relevant_count,
                    "human_count": 0,
                    "n_bull": 0,
                    "n_bear": 0,
                    "n_total": 0,
                    "score": 0.0,
                    "weighted_score": 0.0,
                    "p_est": 0.5,
                    "wilson_lo": 0.0,
                    "wilson_hi": 1.0,
                    "suggested_usd": 0.0,
                    "low_confidence": True
                }

            # Per-comment classification on keywords (we need per-comment labels for weighting)
            ignore_tokens = [ticker_lower]
            labels = [classify_comment_token_label(c["body"], ignore_tokens) for c in human_comments]
            # Unweighted counts
            n_bull = sum(1 for lab in labels if lab == "BULL")
            n_bear = sum(1 for lab in labels if lab == "BEAR")
            n_total = human_count

            # Weighted counts (by comment upvote weight) if requested
            weighted_bull = weighted_bear = total_weight = 0.0
            for lab, c in zip(labels, human_comments):
                w = comment_weight(c.get("score", 0)) if use_weighting else 1.0
                total_weight += w
                if lab == "BULL":
                    weighted_bull += w
                elif lab == "BEAR":
                    weighted_bear += w

            # Compute scores:
            score = (n_bull - n_bear) / max(1, n_total)
            weighted_score = (weighted_bull - weighted_bear) / max(1.0, total_weight)

            # Wilson CI for proportion of bullish among bull+bear mentions
            denom = n_bull + n_bear
            if denom > 0:
                wilson_lo, wilson_hi = wilson_interval(n_bull, denom)
            else:
                # no explicit bull/bear mentions: fallback wide interval
                wilson_lo, wilson_hi = 0.0, 1.0

            # p_est and suggested USD (use unweighted score for p_est mapping for compatibility)
            p_est = score_to_p(score)
            pos_frac = fractional_kelly(p=p_est, b=reward_risk, fraction=kelly_fraction)
            pos_usd = pos_frac * account_size

            low_confidence = (n_total < min_n_total)

            return {
                "ticker": ticker_symbol,
                "collected_mentions": collected_mentions,
                "relevant_count": relevant_count,
                "human_count": human_count,
                "n_bull": n_bull,
                "n_bear": n_bear,
                "n_total": n_total,
                "score": score,
                "weighted_score": weighted_score,
                "p_est": p_est,
                "wilson_lo": wilson_lo,
                "wilson_hi": wilson_hi,
                "suggested_usd": pos_usd,
                "low_confidence": low_confidence
            }

        # ---------- run the sentiment scans for top tickers ----------
        tickers_to_scan = df_rank["Ticker"].tolist()[:min(len(df_rank), max(10, int(top_n)))]
        scan_results = []
        with st.spinner("Scanning tickers for sentiment & computing confidence..."):
            for t in tickers_to_scan:
                try:
                    res = compute_sentiment_for_ticker_enhanced(t, reddit_client, subs, posts_per_sub=posts_per_sub_for_scan, comments_per_post=comments_per_post_for_scan)
                except Exception as e:
                    res = {
                        "ticker": t,
                        "collected_mentions": 0,
                        "relevant_count": 0,
                        "human_count": 0,
                        "n_bull": 0,
                        "n_bear": 0,
                        "n_total": 0,
                        "score": 0.0,
                        "weighted_score": 0.0,
                        "p_est": 0.5,
                        "wilson_lo": 0.0,
                        "wilson_hi": 1.0,
                        "suggested_usd": 0.0,
                        "low_confidence": True
                    }
                    st.write(f"Warning: failed scanning {t}: {e}")
                scan_results.append(res)

        if not scan_results:
            st.warning("No sentiment scan results (maybe reddit client failed).")
            st.stop()

        # Build DataFrame with extra cols
        df_scan = pd.DataFrame(scan_results)
        # Ensure numeric types
        numeric_cols = ["collected_mentions", "relevant_count", "human_count", "n_bull", "n_bear", "n_total", "score", "weighted_score", "p_est", "wilson_lo", "wilson_hi", "suggested_usd"]
        for c in numeric_cols:
            if c in df_scan.columns:
                df_scan[c] = pd.to_numeric(df_scan[c], errors="coerce").fillna(0.0)

        df_scan = df_scan.sort_values(by="collected_mentions", ascending=False).reset_index(drop=True)

        st.subheader("Per-ticker sentiment summary (scanned candidates)")
        # show the most informative columns first
        display_cols = ["ticker", "collected_mentions", "relevant_count", "human_count", "n_bull", "n_bear", "n_total", "score", "weighted_score", "wilson_lo", "wilson_hi", "p_est", "suggested_usd", "low_confidence"]
        st.dataframe(df_scan[display_cols].head(200))

        # Most bullish (by score)
        top_bullish = df_scan.sort_values(by="score", ascending=False).head(10).reset_index(drop=True)
        st.subheader("Top bullish tickers (highest score)")
        st.dataframe(top_bullish[display_cols].head(10))

        # Most bearish (by score)
        top_bearish = df_scan.sort_values(by="score", ascending=True).head(10).reset_index(drop=True)
        st.subheader("Top bearish tickers (lowest score)")
        st.dataframe(top_bearish[display_cols].head(10))

        # Most polarized (largest absolute score) - show weighted variant as well
        df_scan["abs_score"] = df_scan["score"].abs()
        most_polarized = df_scan.sort_values(by="abs_score", ascending=False).head(15).reset_index(drop=True)
        st.subheader("Most polarized tickers (strongest sentiment either way)")
        st.dataframe(most_polarized[display_cols + ["abs_score"]].head(15))

        st.markdown("---")

        # ---------- Quick Analyze selected tickers (summary table) ----------
        st.subheader("Quick Analyze selected tickers")
        selected = st.multiselect("Select tickers to Quick Analyze (sentiment + sizing)", options=df_scan["ticker"].tolist(), default=df_scan["ticker"].tolist()[:3])
        if selected:
            quick_results = []
            for t in selected:
                row = df_scan[df_scan["ticker"] == t].iloc[0]
                quick_results.append((
                    row["ticker"],
                    int(row["collected_mentions"]),
                    int(row["relevant_count"]),
                    int(row["human_count"]),
                    int(row["n_bull"]),
                    int(row["n_bear"]),
                    int(row["n_total"]),
                    float(row["score"]),
                    float(row["weighted_score"]),
                    float(row["wilson_lo"]),
                    float(row["wilson_hi"]),
                    float(row["p_est"]),
                    float(row["suggested_usd"]),
                    bool(row["low_confidence"])
                ))
            summary = pd.DataFrame(quick_results, columns=["Ticker", "Collected", "Relevant", "Human", "Bull", "Bear", "Total", "Score", "WeightedScore", "Wilson_lo", "Wilson_hi", "P_est", "Suggested_$", "LowConf"])
            st.dataframe(summary)

        st.caption(
            "Notes: 'Collected' = unique comments scraped, 'Relevant' = mentions that explicitly referenced the ticker, "
            "'Human' = after bot removal. 'LowConf' marks n_total < min threshold. Wilson interval is on (bull/(bull+bear))."
        )

st.markdown("---")
st.caption("Experimental tool — not financial advice. Use with caution.")