#!/usr/bin/env python3
import os
import time
import json
import logging
import praw
import prawcore
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
import streamlit as st

import os, json
import streamlit as st

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

vertex_client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

from google.auth import default

creds, running_project = default()
print("Authenticating as:", getattr(creds, "service_account_email", creds.__class__.__name__))
print("Using project   :", running_project)
print("Target location :", os.environ.get("GOOGLE_CLOUD_LOCATION"))


refresh_token = os.getenv("REDDIT_REFRESH_TOKEN")
if not refresh_token:
    raise RuntimeError("Please run get_refresh_token() first to populate REDDIT_REFRESH_TOKEN in .env")

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    refresh_token=refresh_token,
    user_agent=USER_AGENT,
    redirect_uri="http://localhost:8080",
    ratelimit_seconds=60,    # PRAW auto-backoff on 429s
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
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments, start=1))
    prompt = f"""
        Below are Reddit comments. Return *only* a JSON array of objects like:
        [{{"index":1,"label":"BOT"|"HUMAN","reason":"..."}}, …]

        {numbered}
        """
    resp = safe_generate_content(prompt, model)
    raw = resp.text.strip()
    # 1) If fenced with ``` or ```json, remove them:
    if raw.startswith("```"):
        # drop leading fence
        raw = raw.split("\n", 1)[1]
        # drop trailing fence
        raw = raw.rsplit("```", 1)[0].strip()
    # 2) Extract the JSON array between the first '[' and the last ']'
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find JSON array in response:\n{raw}")
    json_text = raw[start : end+1]

    # 3) Parse and return
    return json.loads(json_text)
# ─── Main Flow ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    USERNAME = "wsbapp"
    submission = get_latest_submission(USERNAME)
    if not submission:
        print("No submissions found for", USERNAME)
        exit(1)

    # 1) Fetch top‐level comments
    top_level = submission.comments[:50]
    bodies     = [c.body.replace("\n", " ") for c in top_level]

    # 2) Classify them in one batch
    print("=== Bot/Human Classification ===")
    classifications = classify_comments_with_gemini(bodies[:50])

    # 3) Filter out BOT comments
    human_bodies = [
        body for body, cls in zip(bodies[:50], classifications)
        if cls.get("label") != "BOT"
    ]
    print(f"Filtered out {len(bodies[:50]) - len(human_bodies)} bot comments, {len(human_bodies)} left.")

    # 4) Summarize only the human comments
    print("=== Market Consensus (Humans Only) ===")
    consensus = summarize_market_consensus(human_bodies, 50)
    print(consensus)
    st.set_page_config(page_title="Reddit Sentiment Dashboard", layout="wide")
    st.title("Reddit Market Sentiment Analysis")

    # Replace these with your actual data
    classification_results = [
        {"index": i+1, "label": cls["label"], "reason": cls["reason"]}
        for i, cls in enumerate(classifications)
    ]
    summary_lines = [line.strip() for line in consensus.split("\n")]
    # keep only non-empty lines
    summary_bullets = [line for line in summary_lines if line]


    tabs = st.tabs(["👥 Classification", "📈 Summary"])
    with tabs[0]:
        st.subheader("Bot/Human Classification")
        st.table(classification_results)

    with tabs[1]:
        st.subheader("Market Consensus Summary")
        for bullet in summary_bullets:
            st.markdown(f"- {bullet}")
