# ****************************************************************************
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ****************************************************************************


from ...base_algorithm import optimal_parameter
from ..regsd_algorithm import RegSDAlgorithm
from ..regsd_problem import RegSDProblem
from ..regsd_helper import r_int
from math import log2, comb as binomial, ceil, floor
from types import SimpleNamespace

class RegularISDEnum(RegSDAlgorithm):
    def __init__(self, problem: RegSDProblem, **kwargs):
        """Construct an instance of RegularISD-Enum estimator from [ES23]_.

        Args:
            problem (RegSDProblem): An instance of the RegSDProblem class

        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDEnum
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = RegularISDEnum(RegSDProblem(n=100,k=50,w=10))
            >>> A
            RegularISD-Enum estimator for the RegSDProblem with parameters (n, k, w) = (100, 50, 10)
        """
        super(RegularISDEnum, self).__init__(problem, **kwargs)
        n, k, w = self.problem.get_parameters()
        self._name = "RegularISD-Enum"
        self.set_parameter_ranges("p", 0, w)
        self.set_parameter_ranges("ell", 0, n-k+w-1)

        self._default_search_ranges = {key: value.copy() for key,value in self._parameter_ranges.items()}

    @optimal_parameter
    def p(self):
        """Return the optimal parameter p used in the algorithm optimization.
    
        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDEnum
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = RegularISDEnum(RegSDProblem(n=100,k=50,w=10))
            >>> A.p()
            4
        """
        return self._get_optimal_parameter("p")

    @optimal_parameter
    def ell(self):
        """Return the optimal parameter p used in the algorithm optimization.

        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDEnum
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = RegularISDEnum(RegSDProblem(n=100,k=50,w=10))
            >>> A.ell()
            4
        """
        return self._get_optimal_parameter("ell")

    def _are_parameters_invalid(self, parameters: dict):
        """Return if the parameter set `parameters` is invalid."""
        n, k, w = self.problem.get_parameters()
        par = SimpleNamespace(**parameters)
        k_prime = k - w
        v = (k_prime + par.ell) / w
        b = n/w
        if w / 2 < par.p / 2 or v / b >= 1:
            return True
        return False
    def _valid_choices(self):
        """Generator yielding new sets of valid parameters.
    
        Based on the `_parameter_ranges` and already set parameters in `_optimal_parameters`.
        """
        new_ranges = self._fix_ranges_for_already_set_parameters()

        n, k, w = self.problem.get_parameters()
        for p in range(new_ranges["p"]["min"] + new_ranges["p"]["min"] % 2,
                       min(w, new_ranges["p"]["max"]) + 1, 2):
            for ell in range(new_ranges["ell"]["min"], min(n-k+w-1, new_ranges["ell"]["max"]) + 1):
                indices = {"p": p, "ell": ell}
                if not self._are_parameters_invalid(indices):
                    yield indices

    def _find_optimal_parameters(self):
        from ..reliable import balanced_regular
        if (self._optimal_parameters or self._memory_access != 0
                or self.problem.memory_bound != float("inf")
                or self._parameter_ranges != self._default_search_ranges):
            return super()._find_optimal_parameters()
        n, k, w = self.problem.get_parameters()
        self.search_report = balanced_regular(n, k, w, depth=1)
        self._optimal_parameters.update(self.search_report["best"]["parameters"])

    def _compute_time_and_memory_complexity(self, parameters: dict):
        """Return the time complexity of the algorithm for a given set of parameters.
    
        Args:
            parameters (dict): Dictionary including the parameters.
        """
        from ..reliable import balanced_point
        n, k, w = self.problem.get_parameters()
        time, memory = balanced_point(n,k,w,parameters,depth=1)
        return time-log2(n), memory-log2(n)
