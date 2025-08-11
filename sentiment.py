#!/usr/bin/env python3
import os
import time
import json
import praw
import prawcore
from google import genai
from google.genai.errors import ClientError
import streamlit as st
from google.genai.types import Schema, Type  # <-- schema types
from typing import List, Dict
import pandas as pd
import streamlit as st
from textwrap import shorten

st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide")

# Write Google key to a temp file
keyfile = "/tmp/sa.json"
with open(keyfile, "w") as f:
    f.write(st.secrets["google"]["service_account_key"])
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keyfile
os.environ["GOOGLE_CLOUD_PROJECT"]      = "reddit-scrapper-468019"
os.environ["GOOGLE_CLOUD_LOCATION"]     = "global"

# Reddit creds
CLIENT_ID     = st.secrets["reddit"]["REDDIT_CLIENT_ID"]
CLIENT_SECRET = st.secrets["reddit"]["REDDIT_CLIENT_SECRET"]
REFRESH_TOKEN = st.secrets["reddit"]["REDDIT_REFRESH_TOKEN"]
USER_AGENT    = st.secrets["reddit"]["REDDIT_USER_AGENT"]
# … then initialize PRAW and GenAI as before …

# ─── Setup ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_clients():
    # Reddit
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        refresh_token=REFRESH_TOKEN,
        user_agent=USER_AGENT,
        redirect_uri="http://localhost:8080",
        ratelimit_seconds=60,
    )

    # Gemini
    vertex_client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
    )

    return reddit, vertex_client

# materialize once and reuse on reruns
reddit, vertex_client = get_clients()

@st.cache_data(ttl=60, show_spinner=False)
def get_bodies(username: str, limit: int = 50, sort: str = "new") -> list[str]:
    """Fetch top-level comments, with caching to avoid refetching every rerun."""
    submission = get_latest_submission(username)
    if not submission:
        return []
    submission.comment_sort = sort
    safe_replace_more(submission.comments, limit=0)
    top_level = submission.comments[:limit]
    return [c.body.replace("\n", " ") for c in top_level]


@st.cache_data(ttl=120, show_spinner=False)
def classify_batch(comments: list[str]) -> list[dict]:
    return classify_comments_with_gemini(comments)

@st.cache_data(ttl=180, show_spinner=False)
def summarize_cached(comments: list[str], chunk_size: int = 50) -> str:
    return summarize_market_consensus(comments, chunk_size=chunk_size)

from datetime import datetime, timezone

def render_post_header(submission, sort: str, limit: int):
    post_url = f"https://reddit.com{submission.permalink}"
    created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).astimezone()
    author = getattr(submission.author, "name", "[deleted]")
    subr   = submission.subreddit.display_name

    st.markdown(f"### 📌 Viewing comments for: [{submission.title}]({post_url})")
    st.caption(
        f"r/{subr} • by u/{author} • {submission.num_comments} comments • "
        f"score {submission.score} • {created:%b %d, %Y %I:%M %p %Z} • "
        f"sort: **{sort}** • showing top-level: **{limit}**"
    )

# ─── Rate-Limit-Safe replace_more ───────────────────────────────────────────────
def safe_replace_more(comments, limit):
    backoff = 1
    while True:
        try:
            comments.replace_more(limit=limit)
            return
        except prawcore.exceptions.TooManyRequests as e:
            wait = getattr(e, "retry_after", backoff)
            time.sleep(wait)
            backoff = min(backoff * 2, 60)
def get_latest_submission_basic(username: str):
    """Fetch the newest submission for a user without expanding comments."""
    user = reddit.redditor(username)
    for post in user.submissions.new(limit=1):
        return post
    return None

# ─── Utility: Safe, retriable generate_content with permission check ─────────────
def safe_generate_content(contents: str, model: str, max_retries=3, backoff=1):
    for attempt in range(1, max_retries+1):
        try:
            return vertex_client.models.generate_content(
                model=model,
                contents=[contents],
                config={"response_modalities": ["TEXT"]},
            )
        except ClientError as e:
            # Extract error code
            error_code = None
            if hasattr(e, 'error') and isinstance(e.error, dict):
                error_code = e.error.get('code')
            elif e.args and isinstance(e.args[0], int):
                error_code = e.args[0]

            if error_code == 403:
                raise RuntimeError(
                    "Permission denied for Gemini generate_content. "
                    "Ensure your service account/project has aiplatform.models.predict access. "
                    "Visit https://console.developers.google.com to update IAM permissions."
                )
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

# ─── Helpers ────────────────────────────────────────────────────────────────────
def get_latest_submission(username: str):
    user = reddit.redditor(username)
    for post in user.submissions.new(limit=1):
        safe_replace_more(post.comments, limit=0)
        return post
    return None

# ─── Summarization via generate_content ─────────────────────────────────────────
def summarize_market_consensus(comments: list[str], chunk_size=20) -> str:
    """
    Summarize Reddit comments in three sections with up to 9 bullets total:
      1. Stocks & Percentages (with rationale)
      2. Key People Mentioned
      3. General Market Consensus (including crypto)
    """
    summaries = []
    model = "gemini-2.5-flash-lite-preview-06-17"

    # 1) Summarize each chunk into three labeled sections
    for i in range(0, len(comments), chunk_size):
        chunk = comments[i : i + chunk_size]
        joined_comments = "\n\n---\n\n".join(chunk)

        prompt = f"""
            Below are Reddit comments about a market event. 
            For these comments, produce **three labeled sections**, each containing up to **3 bullet points** (9 bullets total):

            1. **Stocks & Percentages**  
            – Call out any ticker symbols (e.g., AAPL, TSLA), the percentage moves (e.g., +5%), and the rationale.

            2. **Key People Mentioned**  
            – List any individuals or entities (e.g., CEOs, Fed officials) and what users said about them.

            3. **General Market Consensus**  
            – Summarize overall sentiment on the broader market and crypto.

            Comments:
            {joined_comments}
            """
        resp = safe_generate_content(prompt, model)
        summaries.append(resp.text.strip())

    # 2) Combine all chunk summaries into one final prompt
    combined = "\n\n".join(f"Batch {idx+1}:\n{batch}" 
                           for idx, batch in enumerate(summaries))

    merge_prompt = f"""
        You have the following partial summaries, each in three sections:

        {combined}

        Please **merge and dedupe** these into a single consolidated summary, maintaining the same three sections and producing up to **3 bullets per section** (9 bullets total).
        """
    final_resp = safe_generate_content(merge_prompt, model)
    return final_resp.text.strip()


# ─── Classification via generate_content ────────────────────────────────────────
def classify_comments_with_gemini(comments: list[str]) -> list[dict]:
    model = "gemini-2.5-flash-lite-preview-06-17"

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(comments, start=1))
    prompt = f"""
You are a JSON-only responder. Return exactly a JSON array for the items below.
Format: [{{"index":1,"label":"BOT"|"HUMAN","reason":"..." }}, ...]
No prose, no code fences.

Comments:
{numbered}
"""

    # Enforce JSON output (and shape) from the API
    schema = Schema(
        type=Type.ARRAY,
        items=Schema(
            type=Type.OBJECT,
            properties={
                "index": Schema(type=Type.INTEGER),
                "label": Schema(type=Type.STRING, enum=["BOT", "HUMAN"]),
                "reason": Schema(type=Type.STRING),
            },
            required=["index", "label", "reason"],
        ),
    )

    resp = vertex_client.models.generate_content(
        model=model,
        contents=[prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )

    # Should be pure JSON now; no fences, no extra text
    return json.loads(resp.text)

def attach_comments_to_results(comments: List[str], results: List[Dict]) -> List[Dict]:
    """results items have 1-based 'index' from your prompt. Add the raw comment text."""
    out = []
    for r in results:
        i = r.get("index")
        if isinstance(i, int) and 1 <= i <= len(comments):
            r = {**r, "comment": comments[i-1]}   # 1-based -> 0-based
        else:
            r = {**r, "comment": ""}
        out.append(r)
    return out


#TRUTH FACT CHECKING


@st.cache_data(ttl=120, show_spinner=False)
def fact_check_batch(comments: list[str], use_search: bool = True) -> list[dict]:
    """
    Returns a JSON list of verdicts for each comment with optional live Google Search grounding.
    Shape:
      [{"index":1,"verdict":"TRUE|LIKELY_TRUE|UNSURE|LIKELY_FALSE|FALSE",
        "confidence":0.0..1.0,
        "claims":[{"text":"...", "support":[{"title":"...", "url":"..."}]}],
        "reason":"..."}, ...]
    """
    if not comments:
        return []

    model = "gemini-2.5-flash-lite-preview-06-17"  # higher reasoning; fall back if you prefer flash-lite

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(comments, start=1))
    prompt = f"""
You are a rigorous fact-checker. Identify *factual claims* in each comment and assess truthfulness.
Consider the *current* world state. If evidence is unclear or mixed, use "UNSURE".

Return ONLY a JSON array with one object per comment:
- index: the 1-based index of the comment
- verdict: one of ["TRUE","LIKELY_TRUE","UNSURE","LIKELY_FALSE","FALSE"]
- confidence: number in [0,1]
- claims: array of objects: {{ "text": str, "support": [{{"title":str, "url":str}}...] }}
- reason: brief explanation in plain English

Comments:
{numbered}
"""

    schema = Schema(
        type=Type.ARRAY,
        items=Schema(
            type=Type.OBJECT,
            properties={
                "index": Schema(type=Type.INTEGER),
                "verdict": Schema(type=Type.STRING, enum=[
                    "TRUE","LIKELY_TRUE","UNSURE","LIKELY_FALSE","FALSE"
                ]),
                "confidence": Schema(type=Type.NUMBER),
                "claims": Schema(
                    type=Type.ARRAY,
                    items=Schema(
                        type=Type.OBJECT,
                        properties={
                            "text": Schema(type=Type.STRING),
                            "support": Schema(
                                type=Type.ARRAY,
                                items=Schema(
                                    type=Type.OBJECT,
                                    properties={
                                        "title": Schema(type=Type.STRING),
                                        "url": Schema(type=Type.STRING),
                                    },
                                    required=["url"]
                                )
                            ),
                        },
                        required=["text","support"]
                    )
                ),
                "reason": Schema(type=Type.STRING),
            },
            required=["index","verdict","confidence","claims","reason"],
        ),
    )

    # Prefer search grounding; gracefully fall back if unavailable
    tools_cfg = [{"google_search": {}}] if use_search else []
    try:
        resp = vertex_client.models.generate_content(
            model=model,
            contents=[prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "tools": tools_cfg,  # enables live search grounding if allowed on your project
            },
        )
        return json.loads(resp.text)
    except ClientError:
        # Retry once without search grounding
        resp = vertex_client.models.generate_content(
            model=model,
            contents=[prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        return json.loads(resp.text)


def attach_comments_to_factchecks(comments: list[str], facts: list[dict]) -> list[dict]:
    """Attach raw comment text to each fact-check row by 1-based index."""
    out = []
    for r in facts:
        i = r.get("index")
        r = {**r, "comment": comments[i-1] if isinstance(i, int) and 1 <= i <= len(comments) else ""}
        out.append(r)
    return out


def filter_comments_by_truth(
    comments: list[str],
    factchecks: list[dict],
    allowed: set[str] = frozenset({"TRUE","LIKELY_TRUE"})
) -> list[str]:
    """
    Keep only comments whose verdict is in `allowed`.
    """
    keep = []
    fc_by_idx = {fc.get("index"): fc for fc in factchecks}
    for i, c in enumerate(comments, start=1):
        v = (fc_by_idx.get(i) or {}).get("verdict", "UNSURE")
        if v in allowed:
            keep.append(c)
    return keep

# TICKER TAB

import re
from collections import defaultdict

TICKER_RE = re.compile(r'\$?[A-Z]{1,5}\b')
PCT_RE    = re.compile(r'([+-]?\d{1,3}(?:\.\d+)?)[ ]?%')
PRICE_RE  = re.compile(r'\$([0-9]{1,6}(?:\.[0-9]{1,2})?)')

# Common false positives to ignore when matching bare A-Z tokens as "tickers"
TICKER_STOP = {
    "A", "I", "T", "YOLO", "CEO", "CFO", "FDA", "AI", "DD", "IMO", "ETF", "EPS",
    "GDP", "CPI", "USD", "OTM", "ITM", "ATM", "WSB", "OPEN", "HIGH", "LOW",
}

ACTIONS = {
    "BUY":  re.compile(r'\b(buy|load(?:ing)?|accumulate|long)\b', re.I),
    "SELL": re.compile(r'\b(sell|trim|take profits?|dump)\b', re.I),
    "HOLD": re.compile(r'\b(hold|baghold|diamond hands?)\b', re.I),
    "SHORT":re.compile(r'\b(short|puts?)\b', re.I),
    "COVER":re.compile(r'\b(cover(?:ing)?)\b', re.I),
}
DIRECTION_UP   = re.compile(r'\b(up|rip|pump|moon|surge|spike|green)\b', re.I)
DIRECTION_DOWN = re.compile(r'\b(down|dump|tank|plunge|red|crash)\b', re.I)

def _clean_ticker(tok: str) -> str:
    tok = tok.lstrip('$')
    return tok

def _find_tickers(text: str) -> list[str]:
    found = []
    for m in TICKER_RE.finditer(text):
        t = _clean_ticker(m.group(0))
        if 1 <= len(t) <= 5 and t.isupper() and t not in TICKER_STOP:
            found.append(t)
    # Deduplicate but preserve order
    seen = set(); out=[]
    for t in found:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _detect_action(text: str) -> str:
    for label, rx in ACTIONS.items():
        if rx.search(text):
            return label
    if DIRECTION_UP.search(text):
        return "UP"
    if DIRECTION_DOWN.search(text):
        return "DOWN"
    return "MENTION"

def _first_pct(text: str):
    m = PCT_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def _first_price(text: str):
    m = PRICE_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None

def extract_ticker_events(truthy_pairs: list[tuple[int, str]]) -> list[dict]:
    """
    truthy_pairs: list of (global_index, comment_text) for comments that passed truth filter.
    Returns rows like:
      {"index": int, "ticker": "AAPL", "action": "BUY|SELL|HOLD|SHORT|COVER|UP|DOWN|MENTION",
       "move_pct": float|None, "price": float|None, "context": "..."}
    """
    rows = []
    for idx, text in truthy_pairs:
        tickers = _find_tickers(text)
        if not tickers:
            continue
        action  = _detect_action(text)
        pct     = _first_pct(text)
        price   = _first_price(text)

        for t in tickers:
            rows.append({
                "index": idx,
                "ticker": t,
                "action": action,
                "move_pct": pct,
                "price": price,
                "context": shorten(text, width=180, placeholder="…"),
            })
    return rows

def aggregate_ticker_stats(rows: list[dict]) -> list[dict]:
    """
    Simple aggregates per ticker.
    """
    agg = defaultdict(lambda: {"mentions":0, "avg_move_pct":None, "buy":0,"sell":0,"hold":0,"short":0})
    for r in rows:
        t = r["ticker"]
        agg[t]["mentions"] += 1
        if r.get("move_pct") is not None:
            # running average
            cur = agg[t]["avg_move_pct"]
            n   = agg[t]["mentions"]
            agg[t]["avg_move_pct"] = (r["move_pct"] if cur is None else (cur*(n-1) + r["move_pct"])/n)
        act = r.get("action","MENTION")
        if act == "BUY":   agg[t]["buy"]   += 1
        if act == "SELL":  agg[t]["sell"]  += 1
        if act == "HOLD":  agg[t]["hold"]  += 1
        if act == "SHORT": agg[t]["short"] += 1

    out = []
    for t, v in agg.items():
        out.append({
            "ticker": t,
            "mentions": v["mentions"],
            "avg_move_pct": (None if v["avg_move_pct"] is None else round(v["avg_move_pct"], 2)),
            "BUY": v["buy"],
            "SELL": v["sell"],
            "HOLD": v["hold"],
            "SHORT": v["short"],
        })
    # sort by mentions desc
    out.sort(key=lambda x: (-x["mentions"], x["ticker"]))
    return out

# ─── Main Flow ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.title("Reddit Market Sentiment Analysis")
    # Controls
    USERNAME = st.text_input("Reddit username", "wsbapp")
    LIMIT = st.slider("Top-level comments to fetch", 10, 100, 50, 10)
    SORT = st.selectbox("Comment sort", ["new", "best", "top", "old", "controversial"], index=0)

    cols = st.columns(3)
    with cols[0]:
        analyze_clicked = st.button("🔎 Analyze (uses Gemini)")
    with cols[1]:
        if st.button("🔄 Refresh comments (no Gemini)"):
            st.cache_data.clear()
            st.rerun()
    with cols[2]:
        if st.button("♻️ Clear results"):
            for k in ("classifications", "rows", "consensus", "analysis_ready"):
                st.session_state.pop(k, None)
            st.rerun()

    # 1) Fetch comments (cached; doesn't hit Gemini)
    bodies = get_bodies(USERNAME, limit=LIMIT, sort=SORT)
    if not bodies:
        st.info("No comments found yet. Try a different user or refresh.")
        st.stop()

    # 🔹 Show which post we're analyzing (cheap fetch; no comment expansion)
    submission = get_latest_submission_basic(USERNAME)
    if submission:
        render_post_header(submission, SORT, LIMIT)

    # 2) Only run Gemini when the button is clicked
    if analyze_clicked:
        with st.spinner("Classifying and summarizing with Gemini…"):
            # 1) Classify (HUMAN/BOT)
            classifications = classify_batch(bodies[:LIMIT])  # cached
            rows = attach_comments_to_results(bodies[:LIMIT], classifications)

            # 2) Keep only human comments for fact-checking
            human_indices = [i for i, cls in enumerate(classifications, start=1) if cls.get("label") != "BOT"]
            human_bodies   = [bodies[i-1] for i in human_indices]

            # 3) Fact-check humans (uses Google Search grounding when available)
            factchecks_all = fact_check_batch(human_bodies, use_search=True)  # cached

            # 4) Filter to only truthy comments (and keep original indices)
            allowed_verdicts = {"TRUE", "LIKELY_TRUE"}
            fc_by_idx = {fc.get("index"): fc for fc in factchecks_all}  # 1-based within human_bodies

            truthy_pairs = []   # list[(global_index, text)]
            for j in range(1, len(human_bodies) + 1):
                verdict = (fc_by_idx.get(j) or {}).get("verdict", "UNSURE")
                if verdict in allowed_verdicts:
                    truthy_pairs.append((human_indices[j-1], human_bodies[j-1]))

            truthy_comments = [c for (_idx, c) in truthy_pairs]

            # 5) Summarize only truthy comments
            consensus = (
                summarize_cached(truthy_comments, chunk_size=LIMIT)
                if truthy_comments else "*(No truthy comments to summarize.)*"
            )

            # 6) Build Tickers & Moves (truth-filtered only)
            ticker_rows  = extract_ticker_events(truthy_pairs)   # requires helpers I gave you
            ticker_stats = aggregate_ticker_stats(ticker_rows)

            # 7) Persist for reruns
            st.session_state.classifications = classifications
            st.session_state.rows = rows
            st.session_state.consensus = consensus
            st.session_state.ticker_rows = ticker_rows
            st.session_state.ticker_stats = ticker_stats
            st.session_state.analysis_ready = True

    # 3) Render results if we have them
    if st.session_state.get("analysis_ready"):
        rows = st.session_state.rows
        df = pd.DataFrame(rows)

        tabs = st.tabs(["👥 Classification", "📊 Tickers & Moves", "📈 Summary"])

        # --- Tab 0: Classification ---
        with tabs[0]:
            st.subheader("Bot/Human Classification")
            labels = st.multiselect("Filter by label", ["HUMAN", "BOT"], default=["HUMAN", "BOT"])
            fdf = df[df["label"].isin(labels)].copy()
            fdf["Comment (preview)"] = fdf["comment"].apply(
                lambda s: shorten(s, width=180, placeholder="…")
            )
            st.dataframe(
                fdf[["index", "label", "reason", "Comment (preview)"]]
                .rename(columns={"index": "Index", "label": "Label", "reason": "Reason"}),
                use_container_width=True,
                hide_index=True,
            )
            st.write("")
            st.subheader("Full comments")
            for r in fdf.sort_values("index").to_dict("records"):
                with st.expander(f"#{r['index']} • {r['label']} • {shorten(str(r.get('reason','')), width=90, placeholder='…')}"):
                    st.write(r["comment"])

        # --- Tab 1: Tickers & Moves (truth-filtered only) ---
        with tabs[1]:
            st.subheader("Tickers & Moves (truth-filtered)")
            tr = st.session_state.get("ticker_rows", [])    # list of event rows
            ts = st.session_state.get("ticker_stats", [])   # per-ticker aggregates

            if not tr:
                st.info("No ticker mentions found in truth-filtered comments.")
            else:
                df_events = pd.DataFrame(tr)
                df_stats  = pd.DataFrame(ts)

                left, right = st.columns(2)
                with left:
                    st.markdown("**Per-Ticker Summary**")
                    if not df_stats.empty:
                        st.dataframe(
                            df_stats.rename(columns={
                                "ticker":"Ticker","mentions":"Mentions","avg_move_pct":"Avg %","BUY":"BUY","SELL":"SELL","HOLD":"HOLD","SHORT":"SHORT"
                            }),
                            use_container_width=True, hide_index=True
                        )
                    else:
                        st.caption("No aggregate stats available.")

                with right:
                    st.markdown("**Most Mentioned (bar)**")
                    if not df_stats.empty and "mentions" in df_stats.columns:
                        # ensure numeric type for chart
                        df_stats["mentions"] = pd.to_numeric(df_stats["mentions"], errors="coerce").fillna(0).astype(int)
                        st.bar_chart(df_stats.set_index("ticker")["mentions"])
                    else:
                        st.caption("Nothing to chart yet.")

                st.write("")
                st.markdown("**Detected Events**")
                st.dataframe(
                    df_events[["index","ticker","action","move_pct","price","context"]]
                        .rename(columns={
                            "index":"Comment #","ticker":"Ticker","action":"Action",
                            "move_pct":"% Move","price":"$ Price","context":"Context"
                        }),
                    use_container_width=True, hide_index=True
                )
                st.caption("Rows derived only from truth-filtered comments.")

        # --- Tab 2: Summary (truth-filtered only) ---
        with tabs[2]:
            st.subheader("Market Consensus (truth-filtered)")
            summary_text = st.session_state.get("consensus", "")
            if not summary_text:
                st.info("No summary available.")
            else:
                summary_bullets = [ln.strip() for ln in summary_text.split("\n") if ln.strip()]
                for bullet in summary_bullets:
                    st.markdown(f"- {bullet}")
    else:
        st.info("Click **Analyze** to run Gemini classification and summarization.")
