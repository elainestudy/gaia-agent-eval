from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

JINA_READER_BASE = "https://r.jina.ai/"
# General pages get a modest budget; Wikipedia's nav chrome and interlanguage link list
# alone can run past 60k characters before the article body even starts, so it needs a
# much larger budget to actually reach real content on long articles.
DEFAULT_MAX_RESPONSE_CHARS = 20000
WIKIPEDIA_MAX_RESPONSE_CHARS = 50000
# Highly distinctive MindTouch/Jina error signature (a raw C# reflection type name) —
# unlike a generic phrase, this essentially never appears in legitimate page content.
DYNAMIC_RENDER_MARKERS = ("property get [Map MindTouch",)
# Keep the alt text (it can carry real information) but drop the noisy CDN URL.
IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
WIKIPEDIA_CONTENT_SELECTOR = "#mw-content-text"


def _is_wikipedia_hostname(hostname: str) -> bool:
    return hostname == "wikipedia.org" or hostname.endswith(".wikipedia.org")


def _targets_private_network(hostname: str) -> bool:
    if not hostname:
        return True
    if hostname == "localhost":
        return True
    try:
        resolved_ip = ipaddress.ip_address(socket.gethostbyname(hostname))
    except (socket.gaierror, ValueError):
        return False
    return (
        resolved_ip.is_private
        or resolved_ip.is_loopback
        or resolved_ip.is_link_local
        or resolved_ip.is_reserved
        or resolved_ip.is_multicast
    )


@tool
def fetch_page(url: str) -> str:
    """Fetch a specific webpage and return its cleaned, readable text content.
    Use this when you already know the target URL (e.g. from a search result) and need to
    read the actual page content instead of relying on a search engine's short snippet."""
    normalized_url = url.strip()
    if not normalized_url.lower().startswith(("http://", "https://")):
        return "FETCH_PAGE_INVALID_URL: provide a full http(s) URL, not a search query."

    hostname = (urlparse(normalized_url).hostname or "").lower()
    if _targets_private_network(hostname):
        return "FETCH_PAGE_BLOCKED: this URL targets a private, local, or reserved network address."
    is_wikipedia = _is_wikipedia_hostname(hostname)

    headers = {"X-Return-Format": "markdown"}
    api_key = os.getenv("JINA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if is_wikipedia:
        headers["X-Target-Selector"] = WIKIPEDIA_CONTENT_SELECTOR

    try:
        response = requests.get(
            JINA_READER_BASE + normalized_url,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            return "FETCH_PAGE_EMPTY: the page returned no readable content."
        if any(marker in text for marker in DYNAMIC_RENDER_MARKERS):
            return (
                "FETCH_PAGE_DYNAMIC_CONTENT: this page renders its content client-side (MindTouch) "
                "and the fetched text is navigation/config, not the actual article body. "
                "Try search_internet to find a cached or mirrored copy of this page's content instead."
            )
        # Inline image URLs (long CDN links) are pure noise for a text QA tool and eat
        # into the truncation budget before real content is reached; keep the alt text.
        text = IMAGE_MARKDOWN_PATTERN.sub(r"\1", text)
        max_chars = WIKIPEDIA_MAX_RESPONSE_CHARS if is_wikipedia else DEFAULT_MAX_RESPONSE_CHARS
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[TRUNCATED: page content exceeds character limit]"
        return text
    except Exception as exc:
        return f"FETCH_PAGE_FAILED: {exc}"
