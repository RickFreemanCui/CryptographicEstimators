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
from math import log2, comb as binomial, ceil, floor, inf
from types import SimpleNamespace


class RegularISDRep(RegSDAlgorithm):
    def __init__(self, problem: RegSDProblem, **kwargs):
        """Construct an instance of RegularISD-Enum estimator in depth 3 from [ES23]_.

        Args:
            problem (RegSDProblem): An instance of the RegSDProblem class

        Examples:
        >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDRep
        >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
        >>> A = RegularISDRep(RegSDProblem(n=100,k=50,w=10))
        >>> A
        RegularISD-Rep estimator for the RegSDProblem with parameters (n, k, w) = (100, 50, 10)
        """
        super(RegularISDRep, self).__init__(problem, **kwargs)
        self._name = "RegularISD-Rep"
        n, k, w = self.problem.get_parameters()

        self.set_parameter_ranges("p", 0, w)
        self.set_parameter_ranges("ell", 0, n - k + w - 1)
        self.set_parameter_ranges("eps_x", 0, w)
        self.set_parameter_ranges("eps_y", 0, w)

        self._default_search_ranges = {key: value.copy() for key,value in self._parameter_ranges.items()}

    @optimal_parameter
    def p(self):
        """Return the optimal parameter p used in the algorithm optimization.

        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDRep
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = RegularISDRep(RegSDProblem(n=300,k=150,w=30))
            >>> A.p()
            8
        """
        return self._get_optimal_parameter("p")

    @optimal_parameter
    def ell(self):
        """Return the optimal parameter ell used in the algorithm optimization.

        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDRep
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = RegularISDRep(RegSDProblem(n=300,k=150,w=30))
            >>> A.ell()
            22
        """
        return self._get_optimal_parameter("ell")

    @optimal_parameter
    def eps_x(self):
        """Return the optimal parameter eps_x used in the algorithm optimization.

        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDRep
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = RegularISDRep(RegSDProblem(n=300,k=150,w=30))
            >>> A.eps_x()
            0
        """
        return self._get_optimal_parameter("eps_x")

    @optimal_parameter
    def eps_y(self):
        """Return the optimal parameter eps_y used in the algorithm optimization.

        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDRep
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = RegularISDRep(RegSDProblem(n=300,k=150,w=30))
            >>> A.eps_y()
            0
        """
        return self._get_optimal_parameter("eps_y")

    def _are_parameters_invalid(self, parameters: dict):
        """Return if the parameter set `parameters` is invalid."""
        n, k, w = self.problem.get_parameters()
        par = SimpleNamespace(**parameters)
        k_prime = k - w
        v = (k_prime + par.ell) / w
        b = n / w
        if w / 2 - par.p / 2 < par.eps_x / 2 or w / 2 - par.p / 4 - par.eps_x / 2 < par.eps_y / 2 \
                or w / 2 < par.p / 2 or v == 0 or v >= b:
            return True
        return False

    def _valid_choices(self):
        """Generator yielding new sets of valid parameters.
    
        Based on the `_parameter_ranges` and already set parameters in `_optimal_parameters`.
        """
        new_ranges = self._fix_ranges_for_already_set_parameters()

        n, k, w = self.problem.get_parameters()
        for p in range((new_ranges["p"]["min"]+3)//4*4, min(w,new_ranges["p"]["max"])+1, 4):
            for eps_x in range((new_ranges["eps_x"]["min"]+1)//2*2, min(w-p,new_ranges["eps_x"]["max"])+1, 2):
                p_x = p//2 + eps_x
                if p_x % 4:
                    continue
                for eps_y in range((new_ranges["eps_y"]["min"]+1)//2*2, min(w-p_x,new_ranges["eps_y"]["max"])+1, 2):
                    for ell in range(new_ranges["ell"]["min"], min(n-k+w-1,new_ranges["ell"]["max"])+1):
                        indices = {"p":p,"ell":ell,"eps_x":eps_x,"eps_y":eps_y}
                        if not self._are_parameters_invalid(indices):
                            yield indices

    def _find_optimal_parameters(self):
        from ..reliable import balanced_regular
        if (self._optimal_parameters or self._memory_access != 0
                or self.problem.memory_bound != float("inf")
                or self._parameter_ranges != self._default_search_ranges):
            return super()._find_optimal_parameters()
        n, k, w = self.problem.get_parameters()
        self.search_report = balanced_regular(n, k, w, depth=3)
        self._optimal_parameters.update(self.search_report["best"]["parameters"])

    def _compute_time_and_memory_complexity(self, parameters: dict):
        """Return the time complexity of the algorithm for a given set of parameters.
    
        Args:
            parameters (dict): Dictionary including the parameters.
        """
        from ..reliable import balanced_point
        n, k, w = self.problem.get_parameters()
        time, memory = balanced_point(n,k,w,parameters,depth=3)
        return time-log2(n), memory-log2(n)
