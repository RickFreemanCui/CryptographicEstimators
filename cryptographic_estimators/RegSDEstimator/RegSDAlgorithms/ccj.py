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
from math import log2, ceil, inf


def _log2_sum_exp2(values):
    finite_values = [value for value in values if value != -inf]
    if not finite_values:
        return -inf
    max_value = max(finite_values)
    if max_value == inf:
        return inf
    return max_value + log2(sum(2 ** (value - max_value) for value in finite_values))


class CCJ(RegSDAlgorithm):
    def __init__(self, problem: RegSDProblem, **kwargs):
        """Construct an instance of CCJ estimator from [CCJ23]_.

        (concrete formulas taken from [ES23]_)

        Args:
            problem (RegSDProblem): an instance of the RegSDProblem class

        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import CCJ
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = CCJ(RegSDProblem(n=100,k=50,w=10))
            >>> A
            CCJ estimator for the RegSDProblem with parameters (n, k, w) = (100, 50, 10)
        """
        super(CCJ, self).__init__(problem, **kwargs)
        n, k, w = self.problem.get_parameters()
        self._name = "CCJ"
        self.set_parameter_ranges("ell", 0, n - k)

    @optimal_parameter
    def ell(self):
        """Return the optimal parameter ell used in the algorithm optimization.

        Examples:
            >>> from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import CCJ
            >>> from cryptographic_estimators.RegSDEstimator import RegSDProblem
            >>> A = CCJ(RegSDProblem(n=100,k=50,w=10))
            >>> A.ell()
            10
        """
        return self._get_optimal_parameter("ell")

    def _valid_choices(self):
        """Generator yielding new sets of valid parameters.
    
        Yields new sets of valid parameters based on the `_parameter_ranges` and
        already set parameters in `_optimal_parameters`.
        """
        new_ranges = self._fix_ranges_for_already_set_parameters()

        n, k, w = self.problem.get_parameters()

        for ell in range(new_ranges["ell"]["min"], min(n-k,new_ranges["ell"]["max"]) + 1):
            yield {"ell": ell}

    def _compute_time_and_memory_complexity(self, parameters: dict):
        """Return the time complexity of the algorithm for a given set of parameters.
    
        Args:
            parameters (dict): Dictionary including the parameters.
        """
        n, k, w = self.problem.get_parameters()
        ell = parameters["ell"]
        k_tilde = k - (1 - (ell + k) / n) / (1 - w / n) * w
        if ell > (n - k_tilde):
            return inf, inf

        # Cost of one iteration, computed in log-domain to avoid overflow for
        # large RegSD instances.
        log_L = log2(n / w) * (w * (k_tilde + ell) / (2 * n))
        log_num_coll = 2 * log_L - ell

        log_terms = [1 + log_L, log_num_coll]
        if n > k_tilde:
            log_terms.append(2 * log2(n - k_tilde))

        # overall cost
        time = _log2_sum_exp2(log_terms)

        return time, log_L
