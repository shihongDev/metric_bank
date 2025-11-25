
# Metric Bank v0 – Design Doc (Draft)

**Status:** Draft
**Owner:** Shihong Liu
**Version:** v0.1
**Scope:** Core SDK, metric registry, rule-based metric routing, LLM adapter scaffolding

---

## 1. Overview

**Metric Bank** is an open-source evaluation infrastructure for GenAI applications.

The core idea:

> **Developers should not need to understand evaluation theory or choose metrics manually.**
> They wrap their functions with a simple `@eval` decorator, and Metric Bank takes care of:
>
> * selecting suitable metrics for the use case,
> * running evaluations (objective + LLM-as-a-judge),
> * producing structured and human-readable reports.

v0 focuses on:

* A clean, extensible **metric registry** with a standard schema.
* A Python **`@eval` decorator** that records inputs/outputs and orchestrates evaluation.
* A **rule-based metric router** for a small set of common task types.
* A pluggable **LLM adapter** interface so that users can later plug in their own local LLM to assist with use-case analysis and metric selection (v0.1+).

---

## 2. Goals & Non-Goals

### 2.1 Goals (v0)

1. **Standardized Metric Schema**

   * Each metric (e.g., `rag_faithfulness_llm_judge`) is a plugin with metadata and a `compute()` function.
   * Schema defined in code and (optionally) YAML/JSON for easier contribution.

2. **`@eval` Decorator for Python Functions**

   * Wrap a Python function (LLM call, RAG pipeline, MCP tool, etc.).
   * Automatically capture inputs, outputs, latency, and errors.
   * Provide a programmatic `.eval(...)` method to run offline evaluations over datasets.

3. **Support a Small Set of Task Types & Metrics**

   * Task types (v0):

     * `classification`
     * `chat`
     * `rag_qa`
     * `tool_calling` (basic)
   * For each task type, provide 2–4 core metrics (mix of objective and LLM-based).

4. **Rule-Based Metric Selection**

   * Given a task type, automatically choose a default set of metrics.
   * Allow users to override by specifying metrics or presets.

5. **LLM Adapter Abstraction**

   * A generic `LLMClient` protocol.
   * Basic implementation supporting OpenAI-compatible endpoints (including local LLM servers).
   * v0 does **not** require full LLM-based metric recommendation, but the interfaces must be ready for it.

### 2.2 Non-Goals (v0)

* Full web dashboard or UI; a simple CLI and text/JSON output is enough.
* Complex online evaluation / drift detection / alerting.
* Deep integration with LangChain / LlamaIndex / MCP (these are v0.2+).
* Large metric zoo; v0 aims for a small but well-designed core.

---

## 3. High-Level Architecture

### 3.1 Modules

Proposed repo layout:

```text
metric-bank/
  metric_bank/
    __init__.py

    core/
      types.py           # Example, UseCase, MetricResult, Run, etc.
      orchestrator.py    # EvalOrchestrator
      registry.py        # MetricRegistry

    metrics/
      classification/
        accuracy.py
        f1.py
        metadata.yaml
      chat/
        helpfulness_llm_judge.py
        latency_ms.py
      rag/
        answer_quality_llm_judge.py
        faithfulness_llm_judge.py
      tool_calling/
        tool_success_rate.py

    router/
      rule_based.py      # Rule-based metric router for v0
      # llm_based.py     # (scaffolding / stub for v0.1)

    llm_adapters/
      base.py            # LLMClient protocol
      openai_compatible.py

    decorators/
      eval_decorator.py  # @eval implementation

    datasets/
      loaders.py         # dataset loading utilities

    cli/
      main.py            # metric-bank eval config.yaml
  examples/
    rag_qa/
    chat/
    classification/
  docs/
  tests/
```

### 3.2 Data Flow (Offline Evaluation)

1. User wraps a function with `@eval(task="rag_qa")`.
2. User calls `my_func.eval(dataset=..., baseline=...)`.
3. The decorator forwards the call to an `EvalOrchestrator`:

   * Loads dataset to a list of `Example`.
   * Uses **rule-based metric router** to select metrics for this task.
   * Executes the user function on each example, captures input/output/cost/latency.
   * Runs all selected metrics’ `compute()` methods.
4. The orchestrator returns a `Run` object.
5. The SDK provides:

   * `run.to_json()` for machine consumption.
   * `run.to_markdown()` for human-readable report / CLI.

---

## 4. Core Data Models

(All type definitions live in `metric_bank/core/types.py`.)

### 4.1 `Example`

Represents one evaluation example.

```python
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

@dataclass
class Example:
    id: str
    inputs: Dict[str, Any]         # function inputs
    output: Optional[Any] = None   # model output (filled after run)
    reference: Optional[Any] = None
    context: Optional[Any] = None  # e.g., RAG documents
    metadata: Dict[str, Any] = None
```

### 4.2 `UseCase`

Represents the user’s use case / problem type.

```python
@dataclass
class UseCase:
    task_type: str                  # "rag_qa", "chat", "classification", ...
    io_schema: Dict[str, Any]       # description of input/output fields
    has_reference: bool
    latency_sensitive: bool
    safety_critical: bool
    cost_sensitive: bool
    domain: str                     # "finance", "general", ...
    notes: Optional[str] = None
```

In v0, `UseCase` is mainly provided by:

* `@eval(task="...")` parameter, plus
* optional manual config (future: LLM-assisted inference).

### 4.3 `MetricMetadata` & `MetricResult`

```python
@dataclass
class MetricMetadata:
    id: str                         # e.g. "rag_faithfulness_llm_judge"
    name: str
    task_types: List[str]           # applicable task types
    metric_type: str                # "objective" | "llm_judge" | "system"
    requires: Dict[str, bool]       # {"reference": True, "context": True, ...}
    cost_estimate: str              # "low" | "medium" | "high"
    description: str
```

```python
@dataclass
class MetricResult:
    metric_id: str
    score: float                    # [0, 1] by convention where possible
    details: Dict[str, Any]         # e.g. per-example scores, rationales
```

### 4.4 `Run`

Represents a single eval run.

```python
@dataclass
class Run:
    id: str
    use_case: UseCase
    metrics: List[MetricMetadata]
    metric_results: List[MetricResult]
    examples: List[Example]
    baseline_name: Optional[str] = None
    candidate_name: Optional[str] = None

    def to_json(self) -> Dict[str, Any]: ...
    def to_markdown(self) -> str: ...
```

---

## 5. Metric Registry & Metric Interface

### 5.1 `Metric` Protocol

Each metric is implemented as a class that conforms to a simple interface.

```python
from typing import Protocol, List

class Metric(Protocol):
    metadata: MetricMetadata

    def compute(self, examples: List[Example]) -> MetricResult:
        ...
```

Metrics live in `metric_bank/metrics/<task>/...`.
Each metric also has a small YAML/JSON metadata file for documentation & discovery, e.g.:

```yaml
id: rag_faithfulness_llm_judge
name: RAG Faithfulness (LLM Judge)
task_types: ["rag_qa"]
metric_type: "llm_judge"
requires:
  reference: false
  context: true
  tool_trace: false
cost_estimate: "high"
description: >
  Checks whether the answer is fully supported by the provided context documents.
```

### 5.2 `MetricRegistry`

`MetricRegistry` provides:

* `register_metric(metric: Metric)`
* `get_metric(metric_id: str) -> Metric`
* `get_metrics_for_task(task_type: str) -> List[Metric]`

In v0, metrics are eagerly registered at import time.

---

## 6. Metric Router (v0 Rule-Based)

### 6.1 Purpose

Given a `UseCase`, choose:

* a set of metric IDs to run;
* for v0, purely rule-based (no LLM logic required).

### 6.2 Interface

```python
class MetricRouter:
    def __init__(self, registry: MetricRegistry):
        self.registry = registry

    def select_metrics(
        self,
        use_case: UseCase,
        explicit_metrics: list[str] | None = None,
        preset: str | None = None,
    ) -> list[Metric]:
        ...
```

### 6.3 Rules (Initial v0 Defaults)

**Task: `classification`**

* Default metrics:

  * `accuracy`
  * `f1_macro`

**Task: `chat`**

* Default metrics:

  * `helpfulness_llm_judge`
  * `latency_ms`

**Task: `rag_qa`**

* Default metrics:

  * `answer_quality_llm_judge`
  * `rag_faithfulness_llm_judge`
  * `latency_ms`

**Task: `tool_calling`**

* Default metrics:

  * `tool_success_rate`
  * `invalid_tool_call_rate`
  * `latency_ms`

The router respects:

* `explicit_metrics`: if provided, use exactly those (validate against registry).
* `preset`: e.g., `"lightweight"` vs `"deep_dive"` (future extension).

LLM-based routing (use local LLM to choose metrics) will be added in `router/llm_based.py` in v0.1+ but the v0 API should be compatible.

---

## 7. LLM Adapter

### 7.1 Motivation

Some metrics require LLM-as-a-judge (e.g., helpfulness, faithfulness).
Metric Bank must support **user-provided local LLMs** via a common interface.

### 7.2 `LLMClient` Protocol

Defined in `llm_adapters/base.py`:

```python
from typing import Protocol, List, Dict

class LLMClient(Protocol):
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        returns: assistant content (str)
        """
        ...
```

### 7.3 OpenAI-Compatible Implementation

In `llm_adapters/openai_compatible.py`:

```python
class OpenAICompatibleLLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        ...

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # call POST {base_url}/chat/completions
        ...
```

This allows:

* Local vLLM / Ollama / LM Studio that expose an OpenAI-style API.
* Remote providers that also follow this standard.

### 7.4 Config

Users can configure the LLM client via:

**Python:**

```python
from metric_bank.llm_adapters import OpenAICompatibleLLMClient
from metric_bank.core import set_llm_client

client = OpenAICompatibleLLMClient(
    base_url="http://localhost:11434/v1",
    api_key="dummy",
    model="qwen2.5-7b-instruct"
)

set_llm_client(client)
```

**YAML (used by CLI in future):**

```yaml
llm:
  type: "openai_compatible"
  base_url: "http://localhost:11434/v1"
  api_key: "dummy"
  model: "qwen2.5-7b-instruct"
```

Metrics that need an LLM-judge will call `LLMClient.chat()` internally.

---

## 8. `@eval` Decorator & SDK API

### 8.1 Decorator Signature

In `decorators/eval_decorator.py`:

```python
from typing import Optional, Callable, Any, List

def eval(
    task: Optional[str] = None,
    auto_detect_task: bool = False,   # v0: reserved, not implemented
    mode: str = "offline",            # "offline" | "online" (v0 only offline)
    metrics: Optional[List[str]] = None,
    metrics_preset: Optional[str] = None,
):
    """
    Decorate a function to make it evaluable via Metric Bank.

    The decorated function will gain an `.eval()` method for offline evaluation.
    """
    ...
```

### 8.2 Decorator Behavior (v0)

* Wraps the user function without changing its call semantics.
* Attaches an `.eval()` method:

```python
@eval(task="rag_qa")
def answer_question(query: str, context_docs: list[str]) -> str:
    ...

# Offline evaluation
run = answer_question.eval(
    dataset="examples/rag_qa/dev.jsonl",
    baseline=None,
)
print(run.to_markdown())
```

### 8.3 `.eval(...)` Method

Proposed signature:

```python
def eval(
    self,
    dataset: str | list[Example] | Any,
    baseline: Optional[str] = None,
    use_case_overrides: Optional[dict] = None,
    metrics: Optional[List[str]] = None,
    metrics_preset: Optional[str] = None,
) -> Run:
    ...
```

Responsibilities:

1. Build `UseCase` from:

   * `task` argument;
   * optional `use_case_overrides`.
2. Load dataset:

   * If `dataset` is a path: use dataset loaders.
   * If a list of `Example`: use directly.
3. Use `MetricRouter` to select metrics.
4. For each example:

   * Call the wrapped function with `example.inputs`.
   * Record `example.output`, latency, errors, etc.
5. Call `compute()` for each metric.
6. Construct and return a `Run`.

---

## 9. Datasets & Loaders (v0)

Simple utility functions in `datasets/loaders.py`:

* `load_jsonl(path) -> List[Example]`

  * Expect each line to be a JSON object:

    * At minimum: `{"id": "...", "inputs": {...}}`
    * Optional: `reference`, `context`, `metadata`.

For v0, we assume small datasets and in-memory processing.

---

## 10. CLI (Minimal)

A minimal CLI wrapper in `cli/main.py`:

```bash
metric-bank eval config.yaml
```

Example `config.yaml`:

```yaml
entrypoint:
  module: "examples.rag_qa.app"
  function: "answer_question"

task: "rag_qa"
dataset: "examples/rag_qa/dev.jsonl"

llm:
  type: "openai_compatible"
  base_url: "http://localhost:11434/v1"
  api_key: "dummy"
  model: "qwen2.5-7b-instruct"
```

CLI responsibilities:

* Import the entrypoint function.
* Apply `@eval` with given config.
* Run `.eval(...)`.
* Print Markdown report to stdout and optionally save JSON.

---

## 11. v0 Scope Summary

**Included:**

* Core data structures (`Example`, `UseCase`, `MetricMetadata`, `MetricResult`, `Run`).
* `Metric` interface + `MetricRegistry`.
* A handful of metrics for:

  * `classification`: `accuracy`, `f1_macro`
  * `chat`: `helpfulness_llm_judge`, `latency_ms`
  * `rag_qa`: `answer_quality_llm_judge`, `rag_faithfulness_llm_judge`, `latency_ms`
  * `tool_calling`: `tool_success_rate`, `invalid_tool_call_rate`, `latency_ms`
* Rule-based metric router.
* `LLMClient` protocol + OpenAI-compatible implementation.
* `@eval` decorator (offline only) + `.eval()` method.
* Simple dataset loader (JSONL).
* Minimal CLI.

**Deferred to v0.1+:**

* Auto-detect task / use case via LLM.
* Full LLM-based metric recommendation (pre-filter + LLM rerank).
* Online sampling & logging, drift monitoring.
* Integrations with external frameworks (LangChain, LlamaIndex, MCP).
* Web UI/dashboard.

---

## 12. Next Steps (Implementation Plan)

1. **Implement core types & registry**

   * `types.py`, `registry.py`, `Metric` protocol.
2. **Implement base metrics**

   * `classification/accuracy.py`, `classification/f1.py`
   * `chat/helpfulness_llm_judge.py`, `chat/latency_ms.py`
   * `rag/answer_quality_llm_judge.py`, `rag/faithfulness_llm_judge.py`
   * `tool_calling/tool_success_rate.py`
3. **Implement LLM adapter**

   * `LLMClient` protocol + `OpenAICompatibleLLMClient`.
4. **Implement rule-based router**

   * `router/rule_based.py` with mapping from task → default metrics.
5. **Implement `@eval` decorator + `.eval()`**

   * In `decorators/eval_decorator.py` and `core/orchestrator.py`.
6. **Create example projects**

   * `examples/rag_qa`, `examples/chat`, etc.
7. **Add tests & basic docs**

   * At least: unit tests for metrics, decorator, registry, router.
   * Basic usage guide in README.

---