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
# ─── Main Flow ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.title("Reddit Market Sentiment Analysis")

    # Controls (optional)
    USERNAME = "wsbapp"
    LIMIT = 50

    # Manual refresh button
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    # 1) Fetch comments (cached)
    bodies = get_bodies(USERNAME, limit=LIMIT, sort="new")
    if not bodies:
        st.error(f"No submissions/comments found for {USERNAME}")
        raise SystemExit

    # 2) Classify (cached)
    st.subheader("Fetching & classifying…")
    classifications = classify_batch(bodies[:LIMIT])

    # 3) Attach original comment text
    rows = attach_comments_to_results(bodies[:LIMIT], classifications)
    df = pd.DataFrame(rows)

    # 4) Filter out BOTs then summarize (both cached)
    human_bodies = [b for b, cls in zip(bodies[:LIMIT], classifications) if cls.get("label") != "BOT"]
    consensus = summarize_cached(human_bodies, chunk_size=LIMIT)

    # ---------- UI ----------
    tabs = st.tabs(["👥 Classification", "📈 Summary"])

    with tabs[0]:
        st.subheader("Bot/Human Classification")
        labels = st.multiselect("Filter by label", ["HUMAN", "BOT"], default=["HUMAN", "BOT"])
        fdf = df[df["label"].isin(labels)].copy()
        fdf["Comment (preview)"] = fdf["comment"].apply(lambda s: shorten(s, width=180, placeholder="…"))
        st.dataframe(
            fdf[["index", "label", "reason", "Comment (preview)"]]
              .rename(columns={"index": "Index", "label": "Label", "reason": "Reason"}),
            use_container_width=True,
            hide_index=True,
        )

        st.write("")  # spacer
        st.subheader("Full comments")
        for r in fdf.sort_values("index").to_dict("records"):
            with st.expander(f"#{r['index']} • {r['label']} • {shorten(r['reason'], 90, placeholder='…')}"):
                st.write(r["comment"])

    with tabs[1]:
        st.subheader("Market Consensus (Humans Only)")
        summary_bullets = [line.strip() for line in consensus.split("\n") if line.strip()]
        for bullet in summary_bullets:
            st.markdown(f"- {bullet}")
