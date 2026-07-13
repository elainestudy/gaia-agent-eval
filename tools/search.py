from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 实例化原始工具
ddg_search = DuckDuckGoSearchRun()


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
    """当你需要查找实时信息、新闻、汇率、天气或无法直接回答的问题时，使用此工具。"""
    if _looks_like_multi_item_query(query):
        specific_item = _extract_first_candidate_from_broad_query(query)
        if specific_item:
            query = specific_item
        else:
            return (
                "QUERY_TOO_BROAD: this search looks like a multi-item list query. "
                "Search one specific candidate item at a time instead of querying the whole list."
            )

    result = ddg_search.invoke(query)
    return result
