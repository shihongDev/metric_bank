"""Ollama-powered demonstration with @eval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

import requests

from ..decorators.eval_decorator import eval

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = DATA_DIR / "ollama_qa_history.jsonl"

DEFAULT_PROMPTS = [
    {
        "id": "loan_policy",
        "question": "What is the maximum LTV for a jumbo mortgage at Metric Bank?",
        "reference": "Metric Bank caps jumbo LTV at 70% across all regions.",
        "context": "Policy doc v24\n- Jumbo mortgage max LTV 70%\n- Standard mortgage max LTV 80%",
    },
    {
        "id": "savings_rate",
        "question": "Give me the current APY for the Metric Bank high-yield savings product.",
        "reference": "Metric Bank High-Yield Savings earns 4.35% APY as of Nov 2025.",
        "context": "Rates sheet 2025-11\nHigh-yield savings APY: 4.35%\nPremier checking: 0.35%",
    },
    {
        "id": "cd_penalty",
        "question": "What is the early withdrawal penalty on a 12 month CD?",
        "reference": "12M CD penalty equals 90 days of simple interest on the amount withdrawn.",
        "context": "CD policy\n6M: 60 days interest\n12M: 90 days interest",
    },
]

OLLAMA_BASE_URL = os.getenv("MB_OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("MB_OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = float(os.getenv("MB_OLLAMA_TIMEOUT", "120"))


@eval(task="rag_qa", name="example_llm_call_qa")
def example_llm_call_qa(question: str, context: str | None = None) -> str:
    """Forward the question to a local Ollama model."""

    response = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Metric Bank's assistant. Answer using provided policy context.",
                },
                {"role": "user", "content": _build_prompt(question, context)},
            ],
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def collect_history(
    prompts: Iterable[dict] = DEFAULT_PROMPTS,
    output_path: Path = HISTORY_PATH,
) -> Path:
    """Generate a dataset by running the local LLM over provided prompts."""

    records: List[dict] = []
    for prompt in prompts:
        answer = example_llm_call_qa(
            question=prompt["question"], context=prompt.get("context")
        )
        records.append(
            {
                "id": prompt["id"],
                "inputs": {
                    "question": prompt["question"],
                    "context": prompt.get("context"),
                },
                "reference": prompt.get("reference"),
                "context": prompt.get("context"),
                "output": answer,
            }
        )
    output_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return output_path


def run_demo(history_path: Path | None = None):
    """Run evaluation over locally-generated Ollama history."""

    dataset_path = history_path or HISTORY_PATH
    if not dataset_path.exists():
        collect_history(output_path=dataset_path)
    run = example_llm_call_qa.eval(dataset=dataset_path)
    print(run.to_markdown())


def _build_prompt(question: str, context: str | None) -> str:
    prompt = f"Question: {question}\n"
    if context:
        prompt += f"\nContext:\n{context}\n"
    prompt += "\nAnswer in complete sentences."
    return prompt


__all__ = [
    "example_llm_call_qa",
    "run_demo",
    "collect_history",
    "HISTORY_PATH",
    "DEFAULT_PROMPTS",
]

