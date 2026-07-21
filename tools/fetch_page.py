from __future__ import annotations

import ipaddress
import os
import re
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from dateutil import parser as date_parser
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
# MindTouch injects the same block of dozens of LaTeX \newcommand macros into every
# page's math rendering setup, regardless of the article's actual content -- pure
# boilerplate that eats into the truncation budget before real content appears.
LATEX_MACRO_LINE_PATTERN = re.compile(r"\\(?:newcommand|renewcommand|definecolor)\{")
WIKIPEDIA_CONTENT_SELECTOR = "#mw-content-text"
# LibreTexts runs on MindTouch, which renders its real content client-side behind
# navigation/config chrome, same problem Wikipedia has. MindTouch pages carry a
# "skip to main content" accessibility link pointing at this anchor.
LIBRETEXTS_CONTENT_SELECTOR = "#elm-main-content"


def _is_wikipedia_hostname(hostname: str) -> bool:
    return hostname == "wikipedia.org" or hostname.endswith(".wikipedia.org")


def _is_libretexts_hostname(hostname: str) -> bool:
    return hostname.endswith(".libretexts.org")


def _is_wayback_hostname(hostname: str) -> bool:
    return hostname == "web.archive.org"


# Jina exposes a page's publish/update metadata as this header line when the source
# HTML has it. Many "live state" pages (rosters, dashboards) don't set that metadata at
# all and instead render today's actual date straight into the visible page text.
PUBLISHED_TIME_PATTERN = re.compile(r"Published Time:\s*(.+)")
DATE_BANNER_PATTERN = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b"
)
LIVE_PAGE_WARNING_THRESHOLD_DAYS = 2


def _detect_page_date(text: str):
    match = PUBLISHED_TIME_PATTERN.search(text)
    candidate = match.group(1).strip() if match else None
    if not candidate:
        banner_match = DATE_BANNER_PATTERN.search(text)
        candidate = banner_match.group(0) if banner_match else None
    if not candidate:
        return None
    try:
        return date_parser.parse(candidate, fuzzy=True).date()
    except (ValueError, OverflowError):
        return None


def _maybe_prepend_live_page_warning(text: str) -> str:
    page_date = _detect_page_date(text)
    if page_date is None:
        return text
    days_old = abs((datetime.now(timezone.utc).date() - page_date).days)
    if days_old > LIVE_PAGE_WARNING_THRESHOLD_DAYS:
        return text
    return (
        f"NOTE: this page is dated {page_date.isoformat()} (today or very recent) -- it "
        "reflects the page's current live state, not necessarily any earlier point in "
        "time. If the question asks about a state as of a different, earlier date, fetch "
        "this URL's Wayback Machine snapshot for that date instead: "
        "https://web.archive.org/web/YYYYMMDD000000/<this URL>\n\n"
    ) + text


# Jina Reader 403s on web.archive.org (likely bot-blocked), and Wayback snapshots are
# already static HTML, so fetch them directly instead of routing through Jina.
def _html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _strip_latex_macro_boilerplate(text: str) -> str:
    lines = [line for line in text.split("\n") if not LATEX_MACRO_LINE_PATTERN.search(line)]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned)


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

    if _is_wayback_hostname(hostname):
        try:
            response = requests.get(normalized_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if response.status_code == 404:
                return (
                    "FETCH_PAGE_FAILED: no archived snapshot exists for this exact URL path. "
                    "The Wayback Machine does not fuzzy-match URLs -- it only works if the URL "
                    "wrapped after /web/YYYYMMDD000000/ is the exact URL of a page you have "
                    "already fetched or seen in a search result. Do not guess, shorten, or "
                    "reconstruct a different path; reuse that exact known-working URL."
                )
            response.raise_for_status()
            text = _html_to_text(response.text)
            if not text:
                return "FETCH_PAGE_EMPTY: the page returned no readable content."
            text = _maybe_prepend_live_page_warning(text)
            if len(text) > DEFAULT_MAX_RESPONSE_CHARS:
                text = text[:DEFAULT_MAX_RESPONSE_CHARS] + "\n\n[TRUNCATED: page content exceeds character limit]"
            return text
        except Exception as exc:
            return f"FETCH_PAGE_FAILED: {exc}"

    is_wikipedia = _is_wikipedia_hostname(hostname)

    headers = {"X-Return-Format": "markdown"}
    api_key = os.getenv("JINA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if is_wikipedia:
        headers["X-Target-Selector"] = WIKIPEDIA_CONTENT_SELECTOR
    elif _is_libretexts_hostname(hostname):
        headers["X-Target-Selector"] = LIBRETEXTS_CONTENT_SELECTOR

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
        text = _strip_latex_macro_boilerplate(text)
        text = _maybe_prepend_live_page_warning(text)
        max_chars = WIKIPEDIA_MAX_RESPONSE_CHARS if is_wikipedia else DEFAULT_MAX_RESPONSE_CHARS
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[TRUNCATED: page content exceeds character limit]"
        return text
    except Exception as exc:
        return f"FETCH_PAGE_FAILED: {exc}"
