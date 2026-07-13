from __future__ import annotations

import re
from urllib.parse import quote

import requests
from langchain_core.tools import tool


WIKIPEDIA_API_BASE = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {
    "User-Agent": "gaia-agent-eval/1.0 (https://github.com/openai)",
}


def _looks_like_multi_item_query(query: str) -> bool:
    comma_count = query.count(",")
    if comma_count >= 3:
        return True

    parts = [part.strip() for part in query.split(",") if part.strip()]
    return len(parts) >= 4


def _extract_specific_item_query(query: str) -> str | None:
    normalized_query = query.strip()
    patterns = [
        r"^(?:is|are)\s+(.+?)\s+(?:a|an)\s+(?:botanical\s+)?(?:fruit|vegetable)(?:\s+botanically)?\??$",
        r"^(?:is|are)\s+(.+?)\s+(?:a|an)\s+(?:fruit|vegetable)\??$",
        r"^botanical classification of:\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized_query, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate and "," not in candidate:
                return candidate
    return None


def _looks_like_yes_no_classification_query(query: str) -> bool:
    lowered_query = query.strip().lower()
    return (
        lowered_query.startswith("is ")
        or lowered_query.startswith("are ")
        or lowered_query.startswith("does ")
        or lowered_query.startswith("do ")
    ) and ("fruit or vegetable" in lowered_query or "vegetable or fruit" in lowered_query)


def _extract_first_candidate_from_broad_query(query: str) -> str | None:
    normalized_query = query.strip()
    if not normalized_query:
        return None

    candidate_source = normalized_query.split(":", 1)[1] if ":" in normalized_query else normalized_query
    candidate_parts = [part.strip(" \t\n\r .;!?") for part in candidate_source.split(",") if part.strip()]
    if not candidate_parts:
        return None

    candidate = candidate_parts[0].strip()
    return candidate or None


def _truncate(text: str, limit: int = 1500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


@tool
def search_wikipedia(query: str) -> str:
    """Use this for concept definitions, category disambiguation, and stable factual background from Wikipedia."""
    try:
        if _looks_like_yes_no_classification_query(query):
            return (
                "WIKIPEDIA_NOT_SUITABLE: this is a yes/no classification question. "
                "Use search_internet for item-level evidence or concept context instead."
            )

        if _looks_like_multi_item_query(query):
            specific_item = _extract_first_candidate_from_broad_query(query)
            if specific_item:
                query = specific_item
            else:
                return (
                    "QUERY_TOO_BROAD: this Wikipedia query looks like a multi-item list query. "
                    "Search one specific candidate item at a time instead of querying the whole list."
                )

        specific_item_query = _extract_specific_item_query(query)
        if specific_item_query:
            query = specific_item_query

        exact_title = query.strip()
        if exact_title:
            exact_response = requests.get(
                WIKIPEDIA_API_BASE,
                params={
                    "action": "query",
                    "titles": exact_title,
                    "prop": "info",
                    "inprop": "url",
                    "format": "json",
                    "utf8": 1,
                },
                headers=WIKIPEDIA_HEADERS,
                timeout=20,
            )
            exact_response.raise_for_status()
            exact_data = exact_response.json()
            pages = exact_data.get("query", {}).get("pages", {})
            exact_page = None
            for page in pages.values():
                if page.get("missing") is None:
                    exact_page = page
                    break
            if exact_page and exact_page.get("title"):
                title = str(exact_page.get("title", "")).strip()
                summary_response = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}",
                    headers=WIKIPEDIA_HEADERS,
                    timeout=20,
                )
                summary_response.raise_for_status()
                summary_data = summary_response.json()

                extract = str(summary_data.get("extract", "")).strip()
                page_url = str(summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")).strip()
                description = str(summary_data.get("description", "")).strip()

                parts = [f"Title: {title}"]
                if description:
                    parts.append(f"Description: {description}")
                if extract:
                    parts.append(f"Summary: {_truncate(extract)}")
                if page_url:
                    parts.append(f"URL: {page_url}")
                return "\n".join(parts)

        search_response = requests.get(
            WIKIPEDIA_API_BASE,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
                "utf8": 1,
            },
            headers=WIKIPEDIA_HEADERS,
            timeout=20,
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return f"WIKIPEDIA_NO_RESULTS: no Wikipedia page matched query: {query}"

        title = str(search_results[0].get("title", "")).strip()
        if not title:
            return f"WIKIPEDIA_NO_RESULTS: query matched an empty title for: {query}"

        summary_response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}",
            headers=WIKIPEDIA_HEADERS,
            timeout=20,
        )
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        extract = str(summary_data.get("extract", "")).strip()
        page_url = str(summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")).strip()
        description = str(summary_data.get("description", "")).strip()

        parts = [f"Title: {title}"]
        if description:
            parts.append(f"Description: {description}")
        if extract:
            parts.append(f"Summary: {_truncate(extract)}")
        if page_url:
            parts.append(f"URL: {page_url}")
        return "\n".join(parts)
    except Exception as exc:
        return f"WIKIPEDIA_LOOKUP_FAILED: {exc}"
