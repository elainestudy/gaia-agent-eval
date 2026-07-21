from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

# DuckDuckGoSearchRun returns one unstructured text blob where URLs (if present
# at all) are truncated and disconnected from the snippet they belong to, so the
# agent has no real URL to hand to fetch_page. DuckDuckGoSearchResults returns
# structured {title, link, snippet} results with a complete URL per result.
ddg_search = DuckDuckGoSearchResults(output_format="list", num_results=6)


def _looks_like_multi_item_query(query: str) -> bool:
    comma_count = query.count(",")
    if comma_count >= 3:
        return True

    parts = [part.strip() for part in query.split(",") if part.strip()]
    return len(parts) >= 4


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


@tool
def search_internet(query: str) -> str:
    """Use this for real-time information, news, exchange rates, weather, or anything
    that cannot be answered directly. Each result includes a full URL -- pass that URL
    to fetch_page to read the actual page instead of relying on the snippet alone."""
    if _looks_like_multi_item_query(query):
        specific_item = _extract_first_candidate_from_broad_query(query)
        if specific_item:
            query = specific_item
        else:
            return (
                "QUERY_TOO_BROAD: this search looks like a multi-item list query. "
                "Search one specific candidate item at a time instead of querying the whole list."
            )

    results = ddg_search.invoke(query)
    if not results:
        return "NO_RESULTS: the search returned nothing."

    formatted = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "").strip()
        link = item.get("link", "").strip()
        snippet = item.get("snippet", "").strip()
        formatted.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")
    return "\n".join(formatted)
