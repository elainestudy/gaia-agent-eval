import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.fetch_page import fetch_page
from tools.get_youtube_transcript import get_youtube_transcript
from tools.search import search_internet
from tools.search_wikipedia import search_wikipedia
from tools.wolframalpha_query import wolframalpha_query
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Load environment variables from .env file
load_dotenv()

# Verify API Key exists in the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Initialize the model using the stable identifier
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

tools = [search_internet, search_wikipedia, wolframalpha_query, get_youtube_transcript, fetch_page]
llm_with_tools = llm.bind_tools(tools)
tool_registry = {tool.name: tool for tool in tools}
SYSTEM_PROMPT = """
You are a careful QA agent.

Rules:
- Use tools when they can provide evidence.
- If the prompt includes a "Local video analysis summary", treat it as evidence from sampled frames.
- If the prompt includes attachment evidence, prefer the parsed attachment over guessing.
- Separate observation from classification: describe first, then classify, then search only if ambiguous.
- When mapping observations to the task target, use broad category membership rather than only prototype examples.
- If an observed instance is an atypical member of the target category, do not exclude it; resolve with knowledge lookup if needed.
- Only search to disambiguate category membership or fill evidence gaps; do not let search replace direct observation.
- Use Wikipedia for stable definitions, entity/category clarification, and conceptual background.
- Use WolframAlpha for arithmetic, unit conversion, equations, and other quantitative lookup.
- If a tool says NO_TRANSCRIPT_AVAILABLE, do not fabricate transcript-based facts.
- Never guess a numeric answer without evidence.
- If evidence is insufficient, continue searching or say the answer cannot be determined from the available information.
- Prefer short, direct final answers.
""".strip()

OBSERVATION_PROMPT_TEMPLATE = """
Observe the media objectively.
List visible entities, text, actions, and spatial relations.
Do not decide the final answer yet.
Do not force entities into the target category before classification.
""".strip()

CLASSIFICATION_PROMPT_TEMPLATE = """
Map observed entities to the task's target category using broad category membership.
If an instance is an atypical or borderline member of the category, keep it as a candidate instead of excluding it.
If category membership is uncertain, mark it as ambiguous and consider knowledge lookup.
""".strip()

KNOWLEDGE_LOOKUP_PROMPT_TEMPLATE = """
Use external knowledge only when category membership is ambiguous, detector output conflicts with common knowledge, or evidence is insufficient.
Search to clarify classification, not to replace observation.
""".strip()

TOOL_SELECTION_PROMPT_TEMPLATE = """
Tool selection guide:
- Use `search_wikipedia` for stable definitions, taxonomy, category membership, and conceptual clarification.
- Use `wolframalpha_query` for arithmetic, unit conversion, equations, and other quantitative or symbolic computation.
- Use `search_internet` for web evidence, recent information, niche sources, or when Wikipedia and WolframAlpha are not enough.
- Use `fetch_page` when you already know a specific URL (from a search result, the question, or an attachment) and need its actual page text, not just a search snippet. Search results are summaries; use fetch_page to read the source directly, especially for reading-comprehension tasks over a known document (e.g. a textbook page, article, or reference page).
- If the question can be answered from the provided evidence, do not call an external tool.
- Do not use WolframAlpha to guess factual knowledge, and do not use Wikipedia to do arithmetic.
- Use Wikipedia for concept pages and noun-like concepts; use search_internet for yes/no classification questions such as "is X a fruit or vegetable".
- If the question includes a candidate list to classify, verify every listed item one by one before finalizing the answer.
- Never send the full list to `search_wikipedia`; query one specific candidate at a time with a short noun phrase, ideally the bare item name.
- Do not assume an item's status from a generic search over the whole list; search only one specific ambiguous item at a time if needed.
- When the prompt already provides the full candidate list, treat that list as the complete evidence set; do not search for additional candidates or use search to reconstruct the whole answer.
- If `search_wikipedia` returns no useful result for a specific concept, explicitly switch to `search_internet` rather than silently treating Wikipedia as sufficient.
- For single-item disambiguation, search the item name itself or a short contextual phrase; avoid turning it into a broad yes/no question.
- When a search result concerns a related but not identical concept, do not let it override the classification of the current item without direct item-specific support.
- For candidate-list tasks, reconcile the final answer against the full item-by-item checklist before responding.
""".strip()

FINALIZE_PROMPT = """
You already have enough evidence to answer the question.
Stop calling tools and provide the final answer now.
Return only the shortest evidence-based answer.
""".strip()

FINAL_FALLBACK_PROMPT = """
You have used all available tool-call steps and have not yet given a final answer.
Tool calls are no longer available.
Based only on the evidence already gathered in this conversation, give your best-effort answer now.
If the evidence is genuinely insufficient, say so briefly, then still give your best guess.
""".strip()

TASK_PROMPT_TEMPLATE = """
Task question:
{question}

Observation policy:
{observation}

Classification policy:
{classification}

Knowledge lookup policy:
{knowledge_lookup}
""".strip()


def _debug_print(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _tool_call_summary(tool_call: dict[str, Any]) -> str:
    tool_name = tool_call.get("name", "<unknown>")
    tool_args = tool_call.get("args", {})
    return f"name={tool_name}, args={tool_args}"


def _normalize_answer_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text", "")
                if text_value:
                    text_parts.append(str(text_value))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts).strip()
    return str(content)


def build_agent_prompt(
    question: str,
    evidence_blocks: list[str] | None = None,
    task_notes: str | None = None,
) -> str:
    evidence_blocks = evidence_blocks or []
    sections: list[str] = [
        TASK_PROMPT_TEMPLATE.format(
            question=question.strip(),
            observation=OBSERVATION_PROMPT_TEMPLATE,
            classification=CLASSIFICATION_PROMPT_TEMPLATE,
            knowledge_lookup=KNOWLEDGE_LOOKUP_PROMPT_TEMPLATE,
        )
    ]

    sections.append(TOOL_SELECTION_PROMPT_TEMPLATE)

    if task_notes:
        sections.append(f"Task notes:\n{task_notes.strip()}")

    clean_evidence = [block.strip() for block in evidence_blocks if block and block.strip()]
    if clean_evidence:
        sections.append("Evidence:\n" + "\n\n".join(clean_evidence))

    sections.append(
        "Answer requirements:\n"
        "- Use evidence first.\n"
        "- If a category is ambiguous, look it up instead of guessing.\n"
        "- Return the shortest answer that is still evidence-based.\n"
        "- For comma-separated list answers, preserve the exact source item strings in the final response.\n"
        "- For candidate-list questions, ensure every source item was considered before finalizing the answer.\n"
        "- Before finalizing any candidate-list answer, mentally check that no source item was skipped and no generic search replaced item-by-item classification.\n"
        "- If the full candidate list is already in the prompt, prefer classifying the provided items directly over any broad search.\n"
        "- Search evidence should support the current item directly; related concepts are not enough to exclude or include an item by themselves."
    )
    return "\n\n".join(sections)


def run_agent(question: str, max_steps: int = 10, verbose: bool = False) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    seen_calls: set[tuple[str, str]] = set()

    _debug_print(verbose, f"[agent] question: {question}")

    for step in range(1, max_steps + 1):
        _debug_print(verbose, f"[agent] step {step}: calling model")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        _debug_print(verbose, f"[agent] step {step}: model content: {response.content!r}")
        _debug_print(verbose, f"[agent] step {step}: tool_calls: {response.tool_calls}")

        if not response.tool_calls:
            _debug_print(verbose, f"[agent] step {step}: final answer reached")
            return _normalize_answer_content(response.content)

        for tool_index, tool_call in enumerate(response.tool_calls, start=1):
            tool_name = tool_call["name"]
            tool = tool_registry.get(tool_name)

            if tool is None:
                raise ValueError(f"Unknown tool requested: {tool_name}")

            _debug_print(verbose, f"[agent] step {step}: tool call {tool_index}: {_tool_call_summary(tool_call)}")

            call_key = (tool_name, json.dumps(tool_call["args"], sort_keys=True, default=str))
            if call_key in seen_calls:
                tool_output = (
                    "ALREADY_TRIED: this exact tool call was already made earlier and did not lead to "
                    "a final answer. Do not repeat it; use a different query, URL, or tool instead."
                )
            else:
                tool_output = tool.invoke(tool_call["args"])
                seen_calls.add(call_key)
            _debug_print(verbose, f"[agent] step {step}: tool output {tool_index}: {tool_output!r}")
            messages.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=str(tool_output),
                )
            )

        if step >= 3:
            messages.append(HumanMessage(content=FINALIZE_PROMPT))

    _debug_print(verbose, "[agent] max_steps reached: forcing a tool-free best-effort answer")
    messages.append(HumanMessage(content=FINAL_FALLBACK_PROMPT))
    fallback_response = llm.invoke(messages)
    _debug_print(verbose, f"[agent] fallback content: {fallback_response.content!r}")
    return _normalize_answer_content(fallback_response.content)











if __name__ == "__main__":
    question = "今天温哥华天气如何？"
    print("正在向 Gemini 提问...")
    answer = run_agent(question)
    print(f"\nGemini 最终回答: {answer}")
