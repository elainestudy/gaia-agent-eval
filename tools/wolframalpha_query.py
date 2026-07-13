from __future__ import annotations

import os

import requests
from langchain_core.tools import tool


WOLFRAMALPHA_API_BASE = "https://api.wolframalpha.com/v1/result"


@tool
def wolframalpha_query(query: str) -> str:
    """Use this for arithmetic, unit conversion, equations, and other quantitative questions."""
    app_id = os.getenv("WOLFRAMALPHA_APP_ID", "").strip()
    if not app_id:
        return "WOLFRAMALPHA_NOT_CONFIGURED: set WOLFRAMALPHA_APP_ID in your environment to enable this tool."

    try:
        response = requests.get(
            WOLFRAMALPHA_API_BASE,
            params={"i": query, "appid": app_id, "output": "JSON"},
            timeout=20,
        )
        if response.status_code == 501:
            return f"WOLFRAMALPHA_NO_RESULT: {response.text.strip()}"
        response.raise_for_status()
        return response.text.strip()
    except Exception as exc:
        return f"WOLFRAMALPHA_QUERY_FAILED: {exc}"
