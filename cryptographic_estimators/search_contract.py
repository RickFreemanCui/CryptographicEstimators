"""Explicit completion semantics for finite-domain concrete optimizations."""
import math
import time

POLICY_VERSION = "finite-domain-rsd-v2"


class SearchIncomplete(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


class SearchBudget:
    """A resource limit must never masquerade as a completed attack minimum."""

    def __init__(self, algorithm, domain, max_evaluations=None, max_seconds=None):
        self.algorithm = algorithm
        self.domain = domain
        self.max_evaluations = max_evaluations
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.evaluations = 0
        self.best = None

    def report(self, status):
        return dict(policy_version=POLICY_VERSION, algorithm=self.algorithm,
                    domain=self.domain, status=status, evaluations=self.evaluations,
                    elapsed_seconds=time.monotonic()-self.started, best=self.best)

    def tick(self, count=1):
        self.evaluations += count
        if ((self.max_evaluations is not None and self.evaluations > self.max_evaluations)
                or (self.max_seconds is not None and time.monotonic()-self.started > self.max_seconds)):
            raise SearchIncomplete("Search budget exhausted: " + self.algorithm,
                                   self.report("budget_exhausted"))

    def consider(self, bits, parameters, memory_bits=None):
        if not math.isfinite(bits):
            raise SearchIncomplete("Non-finite evaluated cost: " + self.algorithm,
                                   self.report("numerical_failure"))
        if self.best is None or bits < self.best["bits"]:
            self.best = dict(bits=float(bits), parameters=dict(parameters), memory_bits=memory_bits)

    def finish(self):
        if self.best is None:
            raise SearchIncomplete("No feasible attack configuration: " + self.algorithm,
                                   self.report("no_feasible_configuration"))
        return self.report("complete")
