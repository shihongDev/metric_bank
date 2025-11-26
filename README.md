# Metric Bank

Early prototype of the Metric Bank SDK described in `design_v0.md`.

## Quick Start (Ollama)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# make sure an Ollama server is running locally
ollama serve --log-level error   # in another terminal

# pull whatever model you want the SDK to use
ollama pull llama3

# generate fresh call history + run eval
python -c "from metric_bank.examples import run_demo; run_demo()"
```

What happens:

1. `example_llm_call_qa` (wrapped by `@eval(task="rag_qa")`) forwards each prompt to the local Ollama model defined by `MB_OLLAMA_MODEL`.
2. `collect_history()` records the actual question/context/reference/output tuples under `metric_bank/examples/data/ollama_qa_history.jsonl`.
3. The orchestrator routes to default RAG metrics (answer quality, faithfulness, BLEU precision, latency).
4. The resulting `Run` is printed in markdown so you can eyeball the scores.

### Environment Overrides

| Variable | Default | Purpose |
| --- | --- | --- |
| `MB_OLLAMA_BASE_URL` | `http://localhost:11434` | Points Metric Bank at your Ollama server |
| `MB_OLLAMA_MODEL` | `llama3` | Name of the model to load via Ollama |
| `MB_OLLAMA_TIMEOUT` | `120` | Seconds to wait on each Ollama request |

### Bringing Your Own Logs

`example_llm_call_qa.eval(dataset=...)` accepts:

- A path to your JSON/JSONL call logs (`{"inputs": {...}, "output": ..., "reference": ...}`).
- A list of `metric_bank.core.types.Example` objects.
- Any iterable of dicts where the decorator can infer `inputs`, `output`, and optional `reference/context`.

Metrics are auto-selected from the registry using the rule-based router, but you can pass `explicit_metrics=["latency_ms"]` or `preset="lightweight"` to customize the run.