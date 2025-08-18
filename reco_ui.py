# reco_ui.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import time
import os

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


# Try to reuse your existing functions. If not found, minimal fallbacks are used.
try:
    from sentiment import get_latest_submission, classify_comments_with_gemini, safe_replace_more
except Exception:
    # Minimal fallback to fetch comments if you don't have sentiment.py in path.
    def get_latest_submission(username):
        raise RuntimeError("Please import your get_latest_submission from your sentiment.py")
    def classify_comments_with_gemini(comments):
        # naive fallback: mark all as HUMAN (no Gemini)
        return [{"index": i+1, "label": "HUMAN", "reason": ""} for i in range(len(comments))]
    
    
    
    

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
            print("Error expanding more comments:", e)
            break

        # small polite pause so we don't burn rate limits
        if pause_between_batches:
            time.sleep(pause_between_batches)

        comments = submission.comments.list()
        # if calling replace_more didn't increase the number of comments, we've exhausted reachable comments
        if len(comments) == prev_len:
            break

    return comments[:max_comments]

# ---------- Helper utilities ----------
BULL_KEYWORDS = {"buy", "bull", "moon", "long", "pump", "call", "rocket", "tendies", "alpha"}
BEAR_KEYWORDS = {"sell", "bear", "short", "dump", "exit", "weak", "stop", "down"}

def simple_sentiment_score(comments):
    """Return (n_bull, n_bear, n_total, score_norm) for list of comment strings."""
    n_bull = n_bear = 0
    for c in comments:
        text = c.lower()
        # simple token check (replaceable with a more sophisticated model)
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

def compute_vwap(df):
    """Compute VWAP for dataframe with columns ['High','Low','Close','Volume']"""
    # Typical price
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    return pv.cumsum() / df["Volume"].cumsum()

def sma(series, window):
    return series.rolling(window).mean()

def fractional_kelly(p, b, fraction=0.5):
    """Return position fraction of bankroll using fractional Kelly.
    p = probability of win (0..1), b = reward/risk (e.g. 1.5 means 1.5:1 reward)
    fraction = fraction of full Kelly to use (e.g., 0.5 = half-Kelly).
    """
    q = 1 - p
    f = 0.0
    if b > 0:
        f_raw = (b * p - q) / b
        f = max(0.0, f_raw) * fraction
    return f

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Reddit→Market Recs", layout="wide")
st.title("Reddit-driven Stock Recommendation (experimental)")
st.markdown(
    """
**Not financial advice.** This is an experimental tool that combines Reddit comment sentiment and simple technical rules to produce *suggestions* only. 
Always do your own research. (See: legal disclaimer.)  
"""
)

# Sidebar inputs
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

run = st.sidebar.button("Analyze now")

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
    top_level = get_up_to_n_comments(submission, max_comments=200, batch=40, pause_between_batches=1.0)
    comments_texts = [c.body.replace("\n", " ") for c in top_level]

    # classify bots (batch)
    st.info("Detecting bot comments (Gemini)...")
    try:
        classifications = classify_comments_with_gemini(comments_texts)
    except Exception as e:
        st.warning("Gemini classification failed — proceeding without bot filtering.")
        classifications = [{"index": i+1, "label": "HUMAN", "reason": ""} for i in range(len(comments_texts))]

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

    # Estimate p (probability of a 'win') from sentiment: map score [-1,1] -> p in [0.1,0.9]
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
                df = df[["High","Low","Close","Volume"]]
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
