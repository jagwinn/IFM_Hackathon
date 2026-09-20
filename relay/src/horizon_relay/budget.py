from .errors import BudgetError
from .settings import Limits


class CloudBudget:
    """Counts cloud calls for one run. One call is always held back for the final synthesis."""

    def __init__(self, limits: Limits):
        self.limits = limits
        self.calls = 0
        self.worker_calls = 0

    def spend(self, operation: str) -> None:
        if not self.limits.cloud_enabled:
            raise BudgetError("Cloud is disabled and no local answer was trusted")
        reserve = 0 if operation in ("synthesize", "answer") else 1  # the call that produces the final answer
        if self.calls >= self.limits.cloud_calls - reserve:
            raise BudgetError("Cloud call budget exhausted (final synthesis reserved)")
        if operation == "work":
            if self.worker_calls >= self.limits.cloud_worker_calls:
                raise BudgetError("Cloud worker budget exhausted")
            self.worker_calls += 1
        self.calls += 1
