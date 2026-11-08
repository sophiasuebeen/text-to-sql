"""The one prompt format used for training, baseline and evaluation.

Changing anything here changes the experiment: bump PROMPT_VERSION and treat
prior baseline/adapter results as incomparable.
"""

from __future__ import annotations

PROMPT_VERSION = "1"

SYSTEM_PROMPT = (
    "You are an expert SQLite assistant. Given a database schema and a question, "
    "write exactly one SQLite SELECT query that answers the question. "
    "Reply with only the SQL query and no explanation."
)

USER_TEMPLATE = "Database schema:\n{schema}\n\nQuestion: {question}"


def build_messages(schema: str, question: str, sql: str | None = None) -> list[dict[str, str]]:
    """Chat messages in the tokenizer's chat-template format.

    With ``sql`` the assistant turn is included (training); without it the
    caller adds the generation prompt (baseline/evaluation).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(schema=schema, question=question.strip())},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql.strip()})
    return messages
