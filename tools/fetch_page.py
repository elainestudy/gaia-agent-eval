from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

JINA_READER_BASE = "https://r.jina.ai/"
# Wikipedia's nav chrome and interlanguage link list alone can run past 60k characters
# before the article body even starts, so the budget has to be generous enough to
# actually reach real content on long articles.
MAX_RESPONSE_CHARS = 50000
# Highly distinctive MindTouch/Jina error signature (a raw C# reflection type name) —
# unlike a generic phrase, this essentially never appears in legitimate page content.
DYNAMIC_RENDER_MARKERS = ("property get [Map MindTouch",)
IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
WIKIPEDIA_CONTENT_SELECTOR = "#mw-content-text"


def _targets_private_network(url: str) -> bool:
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    if hostname.lower() == "localhost":
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
    if _targets_private_network(normalized_url):
        return "FETCH_PAGE_BLOCKED: this URL targets a private, local, or reserved network address."

    headers = {"X-Return-Format": "markdown"}
    api_key = os.getenv("JINA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    hostname = urlparse(normalized_url).hostname or ""
    if hostname.endswith("wikipedia.org"):
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
        # Inline image Markdown (long CDN URLs) is pure noise for a text QA tool and
        # eats into the truncation budget before real content is reached.
        text = IMAGE_MARKDOWN_PATTERN.sub("", text)
        if len(text) > MAX_RESPONSE_CHARS:
            text = text[:MAX_RESPONSE_CHARS] + "\n\n[TRUNCATED: page content exceeds character limit]"
        return text
    except Exception as exc:
        return f"FETCH_PAGE_FAILED: {exc}"
