"""@eval decorator implementation."""

from __future__ import annotations

import functools
import inspect
import time
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

from ..core.context import get_orchestrator
from ..core.types import Example, UseCase
from ..datasets.loaders import load_examples


def eval(
    task: str,
    *,
    name: Optional[str] = None,
    has_reference: bool = True,
    latency_sensitive: bool = False,
    safety_critical: bool = False,
    cost_sensitive: bool = False,
    domain: str = "general",
    io_schema: Optional[dict] = None,
):
    """Decorator entrypoint."""

    def decorator(func: Callable):
        use_case = UseCase(
            task_type=task,
            io_schema=io_schema or _infer_io_schema(func),
            has_reference=has_reference,
            latency_sensitive=latency_sensitive,
            safety_critical=safety_critical,
            cost_sensitive=cost_sensitive,
            domain=domain,
            notes=f"Wrapped by Metric Bank eval for {task}",
        )
        wrapper = EvalWrappedFunction(
            func=func,
            use_case=use_case,
            name=name or func.__name__,
        )
        return functools.update_wrapper(wrapper, func)

    return decorator


class EvalWrappedFunction:
    """Callable wrapper that captures history and exposes `.eval`."""

    def __init__(self, func: Callable, use_case: UseCase, name: str) -> None:
        self._func = func
        self.use_case = use_case
        self.name = name
        self._history: List[Example] = []

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        output = self._func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        example = Example(
            id=f"{self.name}_call_{len(self._history)}",
            inputs=_serialize_inputs(args, kwargs),
            output=output,
            metadata={"source": "live_call"},
            latency_ms=latency_ms,
        )
        self._history.append(example)
        return output

    @property
    def history(self) -> List[Example]:
        return list(self._history)

    def eval(
        self,
        dataset: Union[str, Path, Sequence[Example], Sequence[dict], None] = None,
        *,
        explicit_metrics: Optional[Sequence[str]] = None,
        preset: Optional[str] = None,
        regenerate_missing_outputs: bool = False,
        baseline_name: Optional[str] = None,
        candidate_name: Optional[str] = None,
    ):
        """Run Metric Bank evaluation."""

        if dataset is None:
            examples = list(self._history)
        else:
            examples = load_examples(dataset)
        if regenerate_missing_outputs:
            self._fill_missing_outputs(examples)
        orchestrator = get_orchestrator()
        return orchestrator.run(
            use_case=self.use_case,
            examples=examples,
            explicit_metrics=explicit_metrics,
            preset=preset,
            baseline_name=baseline_name,
            candidate_name=candidate_name or self.name,
        )

    def _fill_missing_outputs(self, examples: List[Example]) -> None:
        for example in examples:
            if example.output is None and example.inputs:
                args, kwargs = _deserialize_inputs(example.inputs)
                example.output = self._func(*args, **kwargs)


def _infer_io_schema(func: Callable) -> dict:
    signature = inspect.signature(func)
    return {
        "inputs": [name for name in signature.parameters.keys()],
        "output": "Any",
    }


def _serialize_inputs(args, kwargs) -> dict:
    payload = {}
    if args:
        payload["_args"] = list(args)
    payload.update(kwargs)
    return payload


def _deserialize_inputs(inputs: dict):
    args = inputs.get("_args", [])
    kwargs = {k: v for k, v in inputs.items() if k != "_args"}
    return args, kwargs

