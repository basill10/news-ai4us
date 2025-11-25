#!/usr/bin/env python3
"""


"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, ValidationError

import requests

try:
    # OpenAI Python SDK (>=1.0)
    from openai import OpenAI
except Exception:
    raise SystemExit("\n[ERROR] Failed to import OpenAI SDK. Install with: pip install openai\n")


# ----------------------------
# ElevenLabs helpers (NEW)
# ----------------------------
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"


def get_elevenlabs_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Resolve the ElevenLabs API key from either:
    - The explicit value passed from the Streamlit sidebar, or
    - ELEVENLABS_API_KEY environment variable.
    """
    return explicit_key or os.getenv("ELEVENLABS_API_KEY")


def elevenlabs_list_voices(api_key: str) -> List[dict]:
    """
    Fetch available voices from ElevenLabs using GET /v1/voices.
    Returns a list of voice dicts with at least 'voice_id' and 'name'.
    """
    url = f"{ELEVENLABS_API_BASE}/voices"
    headers = {"xi-api-key": api_key}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Some docs wrap in {"voices": [...]}, so handle both shapes defensively.
    voices = data.get("voices", data)
    if not isinstance(voices, list):
        voices = []
    return voices


def elevenlabs_tts(api_key: str, voice_id: str, text: str) -> bytes:
    """
    Call ElevenLabs text-to-speech and return MP3 bytes.

    Uses POST /v1/text-to-speech/{voice_id}
    """
    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        # Turbo model is fast & cheap; swap if you prefer another:
        "model_id": "eleven_turbo_v2_5",
        "output_format": "mp3_44100_128",
    }

    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
    resp.raise_for_status()

    chunks: List[bytes] = []
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


# ----------------------------
# Data models
# ----------------------------
class NewsItem(BaseModel):
    title: str = Field(..., description="Headline/title of the news item")
    url: HttpUrl = Field(..., description="Canonical article URL")
    source: str = Field(..., description="Publisher or site name")
    published_date: datetime = Field(..., description="Publication datetime (ISO 8601)")
    summary: str = Field(
        ...,
        description="Neutral but engaging 3–5 sentence summary focused on what happened and why it matters",
    )
    importance: str = Field(
        ...,
        description="Why this matters for ordinary people or specific professions; highlight concrete benefits or risks",
    )
    topics: List[str] = Field(..., description="Tags like 'healthcare', 'law', 'productivity', 'policy', 'models'")
    region: Optional[str] = Field(
        None,
        description="Primary geography or market (e.g. 'US', 'EU', 'India', 'Global', 'MENA'); free-form string",
    )


class NewsBundle(BaseModel):
    timeframe: str
    generated_at: datetime
    items: List[NewsItem]


# ----------------------------
# Text normalization helper
# ----------------------------
def normalize_text(s: str) -> str:
    """Normalize weird unicode characters into clean ASCII-friendly text."""
    if not s:
        return s

    # Normalize unicode (compatibility decomposition + recomposition)
    s = unicodedata.normalize("NFKC", s)

    # Replace curly quotes with straight quotes
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')

    # Replace long/short dashes with hyphens
    s = s.replace("—", "-").replace("–", "-")

    # Replace non-breaking space and similar with regular space
    s = s.replace("\xa0", " ")

    # Remove zero-width characters
    s = re.sub(r"[\u200b\u200c\u200d\u2060]", "", s)

    # Collapse multiple whitespace into a single space
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ----------------------------
# Helper: cap big AI tool / OpenAI-style news to 3
# ----------------------------
BIG_AI_KEYWORDS = [
    "openai",
    "chatgpt",
    "anthropic",
    "claude",
    "google",
    "deepmind",
    "gemini",
    "meta",
    "facebook",
    "llama",
    "microsoft",
    "copilot",
    "azure",
    "amazon",
    "aws",
    "bedrock",
    "x.ai",
    "grok",
    "perplexity",
    "cerebras",
    "databricks",
    "snowflake",
]


def rebalance_news_items(bundle: NewsBundle, max_items: int) -> NewsBundle:
    """
    Enforce:
      - At most 3 items that are primarily about big AI tools / major model vendors.
      - The rest naturally skew toward startups, funding, research, inventions, policy, etc.
    """
    vendor_items: List[NewsItem] = []
    other_items: List[NewsItem] = []

    for item in bundle.items:
        haystack = f"{item.title} {item.source} {' '.join(item.topics)}".lower()
        if any(kw in haystack for kw in BIG_AI_KEYWORDS):
            vendor_items.append(item)
        else:
            other_items.append(item)

    selected_vendor = vendor_items[:3]
    combined = selected_vendor + other_items
    bundle.items = combined[:max_items]
    return bundle


# ----------------------------
# Optional: URL validation to avoid obviously dead links
# ----------------------------
def verify_article_urls(
    items: List[NewsItem],
    max_items: int,
    timeout: int = 8,
) -> List[NewsItem]:
    """
    Make a quick HTTP GET to each URL and keep only those that are 200 OK.
    Also update the URL field to the final redirected URL when possible.

    If nothing validates, caller should fall back to the original list.
    """
    validated: List[NewsItem] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AI-News-Bot/1.0; +https://example.com/bot)"
    }

    for item in items:
        if len(validated) >= max_items:
            break
        try:
            resp = requests.get(str(item.url), timeout=timeout, allow_redirects=True, headers=headers)
            if resp.status_code == 200 and resp.url:
                # Update to the final resolved URL (helps with tracking links or redirects)
                item.url = resp.url  # type: ignore[assignment]
                validated.append(item)
        except Exception:
            # Just skip if we can't reach it
            continue

    return validated


# ----------------------------
# Prompts
# ----------------------------
SYSTEM_INSTRUCTIONS = (
    """
You are an AI news editor for a popular YouTube + newsletter brand.

Audience
--------
• Curious, non-expert general public worldwide.
• Mix of professionals (law, medicine, finance, education, creative fields) and everyday users.
• They care about:
  - How does this affect my work / business / studies / life?
  - What can I do now that I couldn’t do before?
  - What new risks or regulations should I know about?

Scope & Sources
---------------
• Geographic scope: GLOBAL. Include impactful AI stories from any region, but:
  - Prioritize stories with clear, practical implications for everyday people or major professions.
  - It's OK to include some deep-tech stories if you clearly explain "why this matters" in simple terms.

• Use the web_search tool AGGRESSIVELY to:
  1) Find high-impact AI news in the given date range (research, policy, products, startups, funding, lawsuits, major outages).
  2) Prefer original reporting and primary sources; avoid thin rewrites and low-quality blogs.
  3) De-duplicate near-identical coverage.
  4) Exclude opinion-only pieces unless unusually influential.

• Twitter/X & Google Trends (VERY IMPORTANT):
  - Use web_search to look at:
      • Google Trends pages for AI-related topics over the given date range.
      • Twitter/X discussions or coverage around those topics.
  - Use this to:
      • Identify which AI stories are genuinely breaking through to the broader public.
      • Prioritize stories that are being discussed widely, not just in niche research circles.

Diversity constraint (VERY IMPORTANT)
------------------------------------
• Limit stories that are primarily about big AI platforms and tools (e.g., OpenAI, Anthropic, Google/DeepMind, Meta, Microsoft, Amazon,
  and their flagship models like ChatGPT, Gemini, Claude, Llama, Copilot, etc.) to AT MOST 3 items.
• The remaining items MUST focus on:
    - Startups and new AI products from smaller players.
    - Research breakthroughs and new papers with clear practical implications.
    - Funding rounds and acquisitions that change the competitive landscape.
    - Inventions and surprising real-world deployments of AI.
    - Policy, regulation, and lawsuits that affect how AI can be used.

User filters (topics & regions)
-------------------------------
• When the user specifies preferred topics/focus areas, you MUST prioritize stories strongly connected to those themes.
• When the user specifies a preferred region, you MUST prioritize stories whose primary region/market matches that region,
  but you may still include some globally important stories.

Output FORMAT (STRICT)
----------------------
• Output ONLY JSON matching the provided schema (no markdown, no commentary).
• Summaries and importance notes should:
  - Be written in clear, friendly language.
  - Avoid jargon or explain it briefly when necessary.
  - Emphasize "how does this help me?" wherever possible.
    """
).strip()

IMPORTANCE_RUBRIC = (
    """
Score importance by:
• Real-world impact:
    - Does this directly change what people can do (e.g., "lawyers can draft X faster", "doctors can detect Y earlier")?
    - Does it change how businesses or governments operate?
• Breadth:
    - Does it affect many users, companies, or an entire sector?
• Novelty:
    - Is this a breakthrough, a major product launch, or a big regulatory/policy move, not just an incremental update?
• Social & cultural momentum:
    - Is this trending on Twitter/X or in Google Trends?
    - Are lots of people talking about it, beyond just researchers?

In `importance`, write 3–5 sentences explaining:
• Why this story is exciting or worrying.
• Who it helps or harms (e.g., students, doctors, small businesses).
• Any concrete example of what becomes possible because of this story.
    """
).strip()


# ----------------------------
# Core logic: fetch AI news
# ----------------------------
def fetch_ai_news(
    client: OpenAI,
    model: str,
    start: datetime,
    end: datetime,
    max_items: int = 10,
    topics: Optional[List[str]] = None,
    region: Optional[str] = None,
) -> NewsBundle:
    """Call OpenAI Responses API with web_search and parse strict JSON."""

    # Ask the model for a few extra items so we can rebalance & validate URLs.
    requested_items = max_items + 5

    # Build topic + region clauses for the query
    topic_clause = ""
    if topics:
        topic_list_str = ", ".join(topics)
        topic_clause = (
            "You MUST prioritize stories strongly related to the following user-selected topics/focus areas: "
            f"{topic_list_str}. "
        )

    region_clause = ""
    if region:
        # "Any/Global" will be passed as None from the caller, so anything here is a real preference
        region_clause = (
            "You MUST prioritize stories where the primary geography/market matches the user-selected region: "
            f"{region}. You may still include a few globally important stories, but the majority should align "
            "with this region preference. "
        )

    schema = {
        "type": "object",
        "properties": {
            "timeframe": {"type": "string"},
            "generated_at": {"type": "string", "format": "date-time"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string", "format": "uri"},
                        "source": {"type": "string"},
                        "published_date": {"type": "string", "format": "date-time"},
                        "summary": {"type": "string"},
                        "importance": {"type": "string"},
                        "topics": {"type": "array", "items": {"type": "string"}},
                        "region": {"type": ["string", "null"]},
                    },
                    "required": [
                        "title",
                        "url",
                        "source",
                        "published_date",
                        "summary",
                        "importance",
                        "topics",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["timeframe", "generated_at", "items"],
        "additionalProperties": False,
    }

    query = (
        f"Find the {requested_items} most important, engaging, and broadly resonant AI news stories "
        f"published between {start.isoformat()} and {end.isoformat()} (inclusive). "
        "Focus on stories where you can clearly answer: 'How does this help me or people like me?'. "
        + topic_clause
        + region_clause
        + "You MUST obey the diversity constraint: at most 3 stories primarily about big AI platforms "
        "(OpenAI, Anthropic, Google/DeepMind, Meta, Microsoft, Amazon, etc.), and the rest about "
        "startups, funding, research breakthroughs, inventions, policy, and real-world deployments. "
        "VERY IMPORTANT: Only include URLs that point to actual article pages (not placeholders)."
    )

    user_text = (
        query
        + "\n\n"
        + IMPORTANCE_RUBRIC
        + "\n\nOUTPUT FORMAT: Return ONLY JSON matching this schema (no markdown, no prose).\n\n"
        + json.dumps(schema)
    )

    tools_primary = [
        {
            "type": "web_search",
            "search_context_size": "medium",
            "user_location": {"type": "approximate", "country": "US"},
        }
    ]

    tools_fallback = [{"type": "web_search"}]

    def _make_call(tools_payload):
        return client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_text},
            ],
            text={"format": {"type": "text"}},
            reasoning={"effort": "medium"},
            tools=tools_payload,
            store=False,
        )

    try:
        resp = _make_call(tools_primary)
    except Exception as e:
        msg = str(e).lower()
        if (
            "unknown parameter" in msg
            or "unsupported" in msg
            or "invalid_request_error" in msg
            or "bad request" in msg
        ):
            resp = _make_call(tools_fallback)
        else:
            raise

    content = getattr(resp, "output_text", None)
    if not content:
        try:
            content = resp.output[0].content[0].text
        except Exception:
            raise RuntimeError(
                "Unexpected Responses payload; upgrade SDK or include output_text in response."
            )

    try:
        raw = json.loads(content)
        bundle = NewsBundle(**raw)
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(f"Model did not return valid JSON per schema: {e}")

    # Rebalance items to enforce max 3 big AI tool / OpenAI-style stories
    bundle = rebalance_news_items(bundle, max_items=max_items)

    # Validate URLs (optional). If everything fails, fall back to original list.
    validated = verify_article_urls(bundle.items, max_items=max_items)
    if validated:
        bundle.items = validated[:max_items]
    else:
        bundle.items = bundle.items[:max_items]

    return bundle


# ----------------------------
# Newsletter generation (web_search-based)
# ----------------------------
def generate_newsletter(
    client: OpenAI,
    model: str,
    news_item: NewsItem,
    target_words: int = 350,  # <= changed from 900 to 350
) -> str:
    """
    Use the LLM + web_search to research the story on the open web
    (not just the provided URL) and write a full newsletter article.
    """

    system_prompt = (
        "You are an expert newsletter writer for a large, friendly tech media brand. "
        "You write engaging, clear, and practical explainers about AI for a global, non-expert audience. "
        "Your style: conversational, concrete examples, short paragraphs, and very focused on 'how this helps me'. "
        "When needed, you use the web_search tool to read multiple high-quality articles about the story, "
        "especially if the original URL is missing or paywalled."
    )

    user_prompt = f"""
You are writing a newsletter article based on the following AI news story.

======== KNOWN METADATA ========
Headline: {news_item.title}
Source: {news_item.source}
Published date: {news_item.published_date.isoformat()}
Region/Market: {news_item.region or "Global/Unspecified"}
Original URL (may be missing or paywalled): {news_item.url}

High-level summary (from earlier step):
{news_item.summary}

Why it matters (editor notes):
{news_item.importance}

Topics / tags: {", ".join(news_item.topics)}
======== END METADATA ========

RESEARCH INSTRUCTIONS (IMPORTANT)
---------------------------------
1. Use the web_search tool to find:
   - The original article if accessible, OR
   - Other high-quality coverage of the *same* story (matching the headline, source, and topics).
2. Prefer:
   - Reputable news outlets and primary sources.
   - Articles that describe what happened, why it matters, and real-world implications.
3. DO NOT invent details that are not supported by the pages you find.
   - If you are uncertain about a detail, keep it vague or omit it.

WRITING TASK
------------
Write a fully polished newsletter article in MARKDOWN, **no more than {target_words} words**.

Requirements:
- Hook:
  - Start with a short, punchy hook in 2–3 sentences that makes a non-expert care.
- Explain simply:
  - What happened.
  - Why it matters now.
  - Who it helps (e.g., lawyers, doctors, students, small businesses, creators, etc.).
- Be concrete:
  - Give 2–4 specific, real-world examples (e.g., "a small law firm could now...", "a diabetes patient might...").
- Address risks & caveats:
  - Briefly cover any privacy, bias, safety, or job impact concerns if relevant.
- Structure:
  - Use clear section headings (##) and short paragraphs.
  - Use bullet points when listing benefits/risks.
- Tone:
  - Friendly, curious, not hypey.
  - Avoid heavy jargon; if you must use a technical term, explain it in plain language.

OUTPUT:
- Use only MARKDOWN for the final article.
- Do NOT include commentary about your research process or the tools used.
    """.strip()

    tools = [
        {
            "type": "web_search",
            "search_context_size": "high",
            "user_location": {"type": "approximate", "country": "US"},
        }
    ]

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=tools,
        text={"format": {"type": "text"}},
        reasoning={"effort": "medium"},
        store=False,
    )

    content = getattr(resp, "output_text", None)
    if not content:
        try:
            content = resp.output[0].content[0].text
        except Exception:
            raise RuntimeError("Unexpected Responses payload when generating newsletter.")
    return content


# ----------------------------
# NEW: Convert markdown newsletter to a smooth audio script
# ----------------------------
def newsletter_markdown_to_audio_script(
    client: OpenAI,
    model: str,
    newsletter_markdown: str,
) -> str:
    """
    Convert the markdown newsletter into a smooth, conversational audio script.
    The original markdown stays as-is for reading; this is only for TTS.
    """
    system_prompt = (
        "You are an expert at turning written newsletters into spoken audio scripts. "
        "You take markdown articles and rewrite them into a single, flowing narration. "
        "You remove formatting noise (like headings, bullets, links) or weave them into natural sentences. "
        "The output should sound like a human host reading the newsletter out loud, "
        "with smooth transitions between sections."
    )

    user_prompt = f"""
Here is a newsletter article written in MARKDOWN:

---
{newsletter_markdown}
---

TASK:
- Rewrite this as a single, continuous audio script.
- Do NOT include markdown formatting (no #, ##, **, bullet points, or links).
- Turn headings into natural transitions (e.g. "Next, let's talk about...").
- Keep the same information and tone, but make it flow as spoken language.
- Use short, clear sentences that are easy to read aloud.
- Do NOT add new facts; just rephrase and smooth what is already there.
""".strip()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={"format": {"type": "text"}},
        reasoning={"effort": "low"},
        store=False,
    )

    content = getattr(resp, "output_text", None)
    if not content:
        try:
            content = resp.output[0].content[0].text
        except Exception:
            raise RuntimeError("Unexpected Responses payload when generating audio script.")
    return content.strip()


# ----------------------------
# Auth helper
# ----------------------------
def _safe_rerun(st):
    """Handle rerun across Streamlit versions."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.rerun()


def require_login(st):
    """
    Simple username/password login gate.

    - Credentials come from APP_USERNAME / APP_PASSWORD env vars,
      defaulting to 'admin' / 'changeme'.
    - Stores login status in st.session_state["authenticated"].
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = None

    # Already logged in
    if st.session_state["authenticated"]:
        with st.sidebar:
            st.success(f"Logged in as {st.session_state['username']}")
            if st.button("Logout"):
                # Clear auth and other sensitive state
                for key in [
                    "authenticated",
                    "username",
                    "api_key",
                    "bundle",
                    "newsletter",
                    "newsletter_idx",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                _safe_rerun(st)
        return

    # Not logged in yet: show login form
    import os
    valid_username = os.getenv("APP_USERNAME", "admin")
    valid_password = os.getenv("APP_PASSWORD", "changeme")

    with st.sidebar:
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            if username == valid_username and password == valid_password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("Login successful! Reloading...")
                _safe_rerun(st)
            else:
                st.error("Invalid username or password.")

    # Prevent rest of app from running until authenticated
    st.title("📰 AI News of the Week - Caravan Media")
    st.info("Please log in using the sidebar to access the app.")
    st.stop()


# ----------------------------
# Streamlit frontend ONLY
# ----------------------------
def streamlit_main():
    import streamlit as st

    st.set_page_config(page_title="AI News of the Week", layout="wide")

    # Require login before anything else
    require_login(st)

    st.title("📰 AI News of the Week - Caravan Media")

    with st.sidebar:
        st.header("Settings")

        # Sticky API key: stored in session_state instead of hard-coded
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Leave blank to use OPENAI_API_KEY from your environment. "
                 "The value is kept only for this browser session.",
            key="api_key",  # Streamlit will persist this in the session
        )

        # ElevenLabs API key (NEW)
        eleven_api_key = st.text_input(
            "ElevenLabs API Key",
            type="password",
            help="Leave blank to use ELEVENLABS_API_KEY from your environment.",
            key="eleven_api_key",
        )

        # Use whatever model name you actually have access to
        model = st.text_input("Model", value="gpt-5.1")

        today = datetime.now(timezone.utc).date()
        default_start = today - timedelta(days=7)

        start_date = st.date_input("Start date", value=default_start, max_value=today)
        end_date = st.date_input("End date", value=today, min_value=start_date, max_value=today)

        max_items = st.slider("Max stories", min_value=3, max_value=15, value=12)

        # Topic/focus-area filters
        focus_areas = st.multiselect(
            "Focus areas (topics)",
            [
                "policy",
                "healthcare",
                "education",
                "startups",
                "research",
                "creative",
                "coding",
                "funding",
                "safety",
            ],
        )

        # Region filter
        region_choice = st.selectbox(
            "Preferred region",
            [
                "Any / Global",
                "US",
                "EU",
                "UK",
                "India",
                "Pakistan",
                "UAE",
                "China",
                "MENA",
                "Africa",
                "Latin America",
                "APAC (excl. China/India)",
            ],
        )

        fetch_btn = st.button("🔍 Fetch AI News")

    # Normalize region to None when "Any / Global" is selected
    region_filter: Optional[str]
    if region_choice == "Any / Global":
        region_filter = None
    elif region_choice == "APAC (excl. China/India)":
        region_filter = "APAC"
    else:
        region_filter = region_choice

    # Session state
    if "bundle" not in st.session_state:
        st.session_state["bundle"] = None
    if "newsletter" not in st.session_state:
        st.session_state["newsletter"] = None
        st.session_state["newsletter_idx"] = None
    if "newsletter_audio" not in st.session_state:
        st.session_state["newsletter_audio"] = None
    if "eleven_voices" not in st.session_state:
        st.session_state["eleven_voices"] = None
        st.session_state["eleven_voices_key"] = None
    # NEW: store the transcript actually used for audio
    if "newsletter_audio_script" not in st.session_state:
        st.session_state["newsletter_audio_script"] = None

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key

    # Button: start news fetching
    if fetch_btn:
        with st.spinner("Finding the most interesting AI stories..."):
            client = OpenAI(**client_kwargs)
            start_dt = datetime(
                start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc
            )
            end_dt = datetime(
                end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc
            )

            bundle = fetch_ai_news(
                client=client,
                model=model,
                start=start_dt,
                end=end_dt,
                max_items=max_items,
                topics=focus_areas,
                region=region_filter,
            )
            bundle.timeframe = f"{start_dt.date()} → {end_dt.date()}"
            bundle.generated_at = datetime.now(timezone.utc)

            st.session_state["bundle"] = bundle
            st.session_state["newsletter"] = None
            st.session_state["newsletter_idx"] = None
            st.session_state["newsletter_audio"] = None
            st.session_state["newsletter_audio_script"] = None

    bundle: Optional[NewsBundle] = st.session_state.get("bundle")

    # Display fetched headlines + short summaries
    if bundle and bundle.items:
        st.subheader(f"Top AI Stories ({bundle.timeframe})")
        for idx, item in enumerate(bundle.items):
            with st.expander(f"{idx+1}. {item.title}"):
                meta_cols = st.columns([2, 1, 1])
                with meta_cols[0]:
                    st.markdown(
                        f"**Source:** {item.source}  \n"
                        f"**Date:** {item.published_date.strftime('%Y-%m-%d')}"
                    )
                with meta_cols[1]:
                    st.markdown(f"**Region:** {item.region or 'Global/Unspecified'}")
                with meta_cols[2]:
                    st.markdown(f"**Topics:** {', '.join(item.topics)}")

                # Normalize summary text so fonts/characters are uniform
                clean_summary = normalize_text(item.summary)
                st.write(clean_summary)

                st.markdown(f"[Open original article]({item.url})")

                gen_btn = st.button(
                    "✍️ Turn this into a newsletter",
                    key=f"newsletter_btn_{idx}",
                )

                if gen_btn:
                    with st.spinner("Researching this story on the web and writing a newsletter..."):
                        client = OpenAI(**client_kwargs)
                        try:
                            newsletter_md = generate_newsletter(
                                client=client,
                                model=model,
                                news_item=item,
                            )
                        except Exception as e:
                            st.error(
                                "Something went wrong while researching or generating the newsletter.\n\n"
                                f"Details: {e}"
                            )
                        else:
                            st.session_state["newsletter"] = newsletter_md
                            st.session_state["newsletter_idx"] = idx
                            st.session_state["newsletter_audio"] = None  # reset audio
                            st.session_state["newsletter_audio_script"] = None

    # Show generated newsletter
    if st.session_state.get("newsletter"):
        st.markdown("---")
        st.subheader("📣 Generated Newsletter Article")
        st.markdown(
            f"*Based on story #{st.session_state['newsletter_idx'] + 1} above.*"
        )
        st.markdown(st.session_state["newsletter"])

        # Allow user to download the newsletter as a .doc file
        newsletter_text = st.session_state["newsletter"]
        st.download_button(
            "⬇️ Download newsletter (Text)",
            data=newsletter_text,
            file_name="newsletter.txt",
            mime="text/plain",
        )

        # ----------------------------
        # Audio generation with ElevenLabs (NEW)
        # ----------------------------
        st.markdown("### 🔊 Audio version (ElevenLabs)")

        effective_eleven_key = get_elevenlabs_api_key(
            st.session_state.get("eleven_api_key")
        )

        if not effective_eleven_key:
            st.info(
                "To generate audio, add an ElevenLabs API key in the sidebar "
                "or set ELEVENLABS_API_KEY in your environment."
            )
        else:
            # Load voices once per API key
            if (
                st.session_state.get("eleven_voices") is None
                or st.session_state.get("eleven_voices_key") != effective_eleven_key
            ):
                try:
                    voices = elevenlabs_list_voices(effective_eleven_key)
                    st.session_state["eleven_voices"] = voices
                    st.session_state["eleven_voices_key"] = effective_eleven_key
                except Exception as e:
                    st.error(f"Failed to load ElevenLabs voices: {e}")
                    voices = []
                else:
                    voices = st.session_state["eleven_voices"] or []
            else:
                voices = st.session_state.get("eleven_voices") or []

            if not voices:
                st.warning("No ElevenLabs voices found for this API key.")
            else:
                # Prepare dropdown options: "Name (category)" -> voice_id
                options: List[tuple[str, str]] = []
                for v in voices:
                    voice_id = v.get("voice_id")
                    if not voice_id:
                        continue
                    name = v.get("name") or voice_id
                    category = v.get("category") or ""
                    label = f"{name} ({category})" if category else name
                    options.append((label, voice_id))

                labels = [lbl for (lbl, _vid) in options]
                if labels:
                    selected_label = st.selectbox(
                        "Choose a voice",
                        labels,
                        key="eleven_voice_select",
                    )
                    selected_voice_id = dict(options)[selected_label]

                    if st.button("🎧 Generate audio for this newsletter"):
                        with st.spinner("Preparing a readable audio script and generating audio..."):
                            try:
                                # Use OpenAI to turn the markdown newsletter into a smooth audio script
                                client = OpenAI(**client_kwargs)
                                audio_script = newsletter_markdown_to_audio_script(
                                    client=client,
                                    model=model,
                                    newsletter_markdown=st.session_state["newsletter"],
                                )
                                st.session_state["newsletter_audio_script"] = audio_script

                                # Now send the audio-friendly script to ElevenLabs, not the raw markdown
                                audio_bytes = elevenlabs_tts(
                                    effective_eleven_key,
                                    selected_voice_id,
                                    audio_script,
                                )
                            except Exception as e:
                                st.error(f"Failed to generate audio: {e}")
                            else:
                                st.session_state["newsletter_audio"] = audio_bytes

        # If audio present, show player & download
        if st.session_state.get("newsletter_audio"):
            st.audio(st.session_state["newsletter_audio"], format="audio/mpeg")
            st.download_button(
                "⬇️ Download audio (MP3)",
                data=st.session_state["newsletter_audio"],
                file_name="newsletter_audio.mp3",
                mime="audio/mpeg",
            )

        # Optional: show the exact script used for TTS
        if st.session_state.get("newsletter_audio_script"):
            with st.expander("Show transcript used for audio"):
                st.text(st.session_state["newsletter_audio_script"])


if __name__ == "__main__":
    streamlit_main()
