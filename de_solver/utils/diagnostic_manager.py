import time
from typing import Any, Dict


class DiagnosticManager:
    """
    Collects simple run diagnostics such as CPU time, step count,
    conservation metrics, etc.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0
        self.steps: int = 0
        self.records: Dict[str, Any] = {}

    def start(self) -> None:
        """Mark the beginning of a run."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Mark the end of a run and compute elapsed time."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

    def tick(self) -> None:
        """Increment the step counter."""
        self.steps += 1

    def record(self, key: str, value: Any) -> None:
        """Store an arbitrary metric (e.g., mass, energy)."""
        self.records[key] = value

    def summary(self) -> Dict[str, Any]:
        """Return all diagnostics in a single dict."""
        return {
            "steps": self.steps,
            "elapsed_s": round(self.elapsed, 6),
            "records": self.records,
        }

    def __repr__(self) -> str:
        return f"DiagnosticManager(steps={self.steps}, elapsed={self.elapsed:.3f}s)"