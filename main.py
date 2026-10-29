"""
CLI runner for the DocuBot tinker activity.

Supports three modes:
1. Naive LLM generation over all docs (Phase 0)
2. Retrieval only (Phase 1)
3. RAG: retrieval plus LLM generation (Phase 2)
plus a fourth option that runs modes 1-3 on the same query and compares them.
"""

import textwrap

from dotenv import load_dotenv
load_dotenv()

from docubot import DocuBot
from llm_client import GeminiClient
from dataset import SAMPLE_QUERIES


def try_create_llm_client():
    """
    Tries to create a GeminiClient.
    Returns (llm_client, has_llm: bool).
    """
    try:
        client = GeminiClient()
        return client, True
    except RuntimeError as exc:
        print("Warning: LLM features are disabled.")
        print(f"Reason: {exc}")
        print("You can still run retrieval only mode.\n")
        return None, False


def choose_mode(has_llm):
    """
    Asks the user which mode to run.
    Returns "1", "2", "3", "4", or "q".
    """
    print("Choose a mode:")
    if has_llm:
        print("  1) Naive LLM over full docs (no retrieval)")
    else:
        print("  1) Naive LLM over full docs (unavailable, no GEMINI_API_KEY)")
    print("  2) Retrieval only (no LLM)")
    if has_llm:
        print("  3) RAG (retrieval + LLM)")
    else:
        print("  3) RAG (unavailable, no GEMINI_API_KEY)")
    print("  4) Compare all modes side by side (table)")
    print("  q) Quit")

    choice = input("Enter choice: ").strip().lower()
    return choice


def get_query_or_use_samples():
    """
    Ask the user if they want to run all sample queries or a single custom query.

    Returns:
        queries: list of strings
        label: short description of the source of queries
    """
    print("\nPress Enter to run built in sample queries.")
    custom = input("Or type a single custom query: ").strip()

    if custom:
        return [custom], "custom query"
    else:
        return SAMPLE_QUERIES, "sample queries"


def run_naive_llm_mode(bot, has_llm):
    """
    Mode 1:
    Naive LLM generation over the full docs corpus.
    """
    if not has_llm or bot.llm_client is None:
        print("\nNaive LLM mode is not available (no GEMINI_API_KEY).\n")
        return

    queries, label = get_query_or_use_samples()
    print(f"\nRunning naive LLM mode on {label}...\n")

    all_text = bot.full_corpus_text()

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.llm_client.naive_answer_over_full_docs(query, all_text)
        print("Answer:")
        print(answer)
        print()


def run_retrieval_only_mode(bot):
    """
    Mode 2:
    Retrieval only answers. No LLM involved.
    """
    queries, label = get_query_or_use_samples()
    print(f"\nRunning retrieval only mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.answer_retrieval_only(query)
        print("Retrieved snippets:")
        print(answer)
        print()


def run_rag_mode(bot, has_llm):
    """
    Mode 3:
    Retrieval plus LLM generation.
    """
    if not has_llm or bot.llm_client is None:
        print("\nRAG mode is not available (no GEMINI_API_KEY).\n")
        return

    queries, label = get_query_or_use_samples()
    print(f"\nRunning RAG mode on {label}...\n")

    for query in queries:
        print("=" * 60)
        print(f"Question: {query}\n")
        answer = bot.answer_rag(query)
        print("Answer:")
        print(answer)
        print()


COLUMN_WIDTHS = (12, 24, 62)


def shorten(text, limit=240):
    """Collapses whitespace and cuts text to limit characters."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_table(headers, rows, widths=COLUMN_WIDTHS):
    """
    Renders rows as a text table. Cells wrap onto multiple lines
    to fit their column width.
    """
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def render_row(cells):
        wrapped = [textwrap.wrap(cell, w) or [""] for cell, w in zip(cells, widths)]
        height = max(len(lines) for lines in wrapped)
        out = []
        for i in range(height):
            parts = [
                (lines[i] if i < len(lines) else "").ljust(w)
                for lines, w in zip(wrapped, widths)
            ]
            out.append("| " + " | ".join(parts) + " |")
        return out

    lines = [border, *render_row(headers), border]
    for row in rows:
        lines.extend(render_row(row))
        lines.append(border)
    return "\n".join(lines)


def compare_modes_for_query(bot, query, has_llm):
    """
    Runs one query through modes 1, 2, and 3.
    Returns table rows: (mode, sources, answer).
    """
    retrieved = bot.retrieve(query)
    files = list(dict.fromkeys(name for name, _ in retrieved))  # unique, in order
    sources = ", ".join(files) if files else "(none)"
    unavailable = "(unavailable, no GEMINI_API_KEY)"

    if has_llm and bot.llm_client is not None:
        naive = bot.llm_client.naive_answer_over_full_docs(query, bot.full_corpus_text())
        rag = bot.answer_rag(query)
    else:
        naive = rag = unavailable

    return [
        ("1 Naive LLM", "(ignores docs)", shorten(naive)),
        ("2 Retrieval", sources, shorten(bot.answer_retrieval_only(query))),
        ("3 RAG", sources if has_llm else "-", shorten(rag)),
    ]


def run_compare_mode(bot, has_llm):
    """
    Mode 4:
    Runs each query through modes 1, 2, and 3 and prints a comparison table.
    """
    queries, label = get_query_or_use_samples()
    print(f"\nComparing all modes on {label}...\n")

    for query in queries:
        print(f"Question: {query}")
        rows = compare_modes_for_query(bot, query, has_llm)
        print(format_table(("Mode", "Sources", "Answer"), rows))
        print()


def main():
    print("DocuBot Tinker Activity")
    print("=======================\n")

    llm_client, has_llm = try_create_llm_client()
    bot = DocuBot(llm_client=llm_client)

    while True:
        choice = choose_mode(has_llm)

        if choice == "q":
            print("\nGoodbye.")
            break
        elif choice == "1":
            run_naive_llm_mode(bot, has_llm)
        elif choice == "2":
            run_retrieval_only_mode(bot)
        elif choice == "3":
            run_rag_mode(bot, has_llm)
        elif choice == "4":
            run_compare_mode(bot, has_llm)
        else:
            print("\nUnknown choice. Please pick 1, 2, 3, 4, or q.\n")


if __name__ == "__main__":
    main()
