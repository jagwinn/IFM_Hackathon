class RelayError(Exception):
    """An actionable, user-safe orchestration error. Messages never contain credentials or provider bodies."""


class ValidationError(RelayError):
    """A plan or worker result failed its checks."""


class BudgetError(RelayError):
    """A cloud call would exceed the configured limits."""
