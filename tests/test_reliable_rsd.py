import math
import unittest
from cryptographic_estimators.search_contract import SearchBudget, SearchIncomplete
from cryptographic_estimators.RegSDEstimator import RegSDProblem
from cryptographic_estimators.RegSDEstimator.RegSDAlgorithms import RegularISDEnum, RegularISDRep
from cryptographic_estimators.RegSDEstimator.reliable import balanced_regular, estimate
from cryptographic_estimators.SDEstimator.sd_algorithm import SDAlgorithm


class ReliableSearchTests(unittest.TestCase):
    def test_vectorized_regular_search_matches_independent_scalar_enumeration(self):
        for cls, depth in [(RegularISDEnum,1),(RegularISDRep,3)]:
            algo=cls(RegSDProblem(48,24,8))
            brute=math.inf
            for point in algo._valid_choices():
                t,_=algo._compute_time_and_memory_complexity(point)
                brute=min(brute,t+math.log2(48))
            result=balanced_regular(48,24,8,depth)
            self.assertEqual(result['status'],'complete')
            self.assertAlmostEqual(brute,result['best']['bits'],places=10)

    def test_odd_block_count_has_explicit_unequal_half_model(self):
        from cryptographic_estimators.RegSDEstimator.reliable import balanced_point
        for depth in (1,2,3):
            r=balanced_regular(54,27,9,depth)
            scalar=balanced_point(54,27,9,r['best']['parameters'],depth)[0]
            self.assertAlmostEqual(r['best']['bits'],scalar,places=10)
            self.assertIn('floor/ceil',r['domain']['rounding'])

    def test_omitted_balanced_residue_class_is_enumerated(self):
        algo=RegularISDRep(RegSDProblem(3504,2452,292))
        point=dict(p=116,ell=484,eps_x=22,eps_y=0)
        for key,value in point.items():algo.set_parameter_ranges(key,value,value)
        self.assertEqual(list(algo._valid_choices()),[point])
        self.assertAlmostEqual(algo.time_complexity(),375.6276624386437,places=9)

    def test_full_natural_rep_domain_reproduces_large_counterexample(self):
        report=balanced_regular(3504,2452,292,3)
        self.assertAlmostEqual(report['best']['bits'],375.6276624386437,places=9)
        self.assertEqual(report['best']['parameters']['p'],116)

    def test_range_controller_does_not_overwrite_an_earlier_change(self):
        class Fake:
            _adjust_radius=10
            def p(self):return 34
            def l(self):return 100
        f=Fake();f._optimal_parameters_methods=[f.p,f.l]
        f._parameter_ranges={'p':{'min':0,'max':35},'l':{'min':0,'max':500}}
        self.assertFalse(SDAlgorithm._adjust_parameter_ranges(f))
        self.assertEqual(f._parameter_ranges['p']['max'],45)

    def test_exhaustion_and_unsupported_rounding_are_not_certificates(self):
        with self.assertRaises(SearchIncomplete) as caught:
            balanced_regular(3504,2452,292,3,max_evaluations=1)
        self.assertEqual(caught.exception.report['status'],'budget_exhausted')
        with self.assertRaises(SearchIncomplete):balanced_regular(3454,1850,691)
        with self.assertRaises(SearchIncomplete):estimate(96,48,16,max_evaluations=1)

    def test_small_full_portfolio_completes(self):
        result=estimate(96,48,16,max_evaluations=1000000)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(result['algorithms']),8)
        self.assertTrue(all(a['status']=='complete' for a in result['algorithms']))


if __name__=='__main__':unittest.main()
