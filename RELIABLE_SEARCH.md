# Maintained binary RSD search policy

This fork starts from upstream `071cce7a942af423422cb0c6d6af55778c72e4d1`.
The strict entry point is:

```python
from cryptographic_estimators.RegSDEstimator.reliable import estimate
report = estimate(N, k, w, max_seconds=60)
# Or RegSDEstimator(N, k, w).reliable_estimate(max_seconds=60)
```

`status == "complete"` means the declared finite domains in this report were
exhausted, with only cost lower-bound pruning. It is not a security proof, a
certificate covering every decoding algorithm, or a physical gate count.
The policy identifier is `finite-domain-rsd-v2`. Keep complete reports, model
identifiers, and source fingerprints with cached costs. Do not reuse legacy
scalar caches. A resource limit raises `SearchIncomplete` and carries a partial
report; it must not be replaced by another algorithm's higher cost.

## Search repairs

* Regular Enum searches every even `p <= w` and every `0 <= ell < N-k+w`.
* Regular Rep searches `p` in steps of 4, even `eps_x`, `p_x % 4 == 0`, even
  `eps_y`, and the full ell interval. The bounds are `eps_x <= w-p` and
  `eps_y <= w-p_x`; there are no fixed 40/32/20 caps, p<=w/2 cap, or list-size
  window. In particular, `p=4 mod 8, eps_x=2 mod 4` is retained.
* CCJ searches its entire feasible ell interval, using log-domain arithmetic.
* The ordinary SD range controller no longer lets a later unchanged parameter
  overwrite an earlier expansion. Strict `full_domain=True` bypasses this
  heuristic controller, removes the BJMM depth-2 ell window, uses natural
  supported bounds, and includes the omitted Ball-Collision odd-p points.
* Negative BJMM merge-bit counts are rejected explicitly, rather than producing
  huge float powers or an overflow. Exact binomial memoization changes runtime,
  not the formulas. A failed component is not silently omitted by the strict API.

The retained SD portfolio is Ball-Collision, BJMM depths 2/3, and Prange.
The upstream exclusions of other SD algorithms are explicit coverage limits.
Generic table output and heuristic SD searches are not completion certificates.
Only the strict supported binary portfolio has the completion semantics above.

## Model boundaries and rounding

These are modeled log2 bit-operation costs. The strict API fixes unlimited
memory, memory-access cost 0, and applies the vector-length conversion once.
The regular models follow ES24, Section 4.3 and Appendix A.1:
<https://eprint.iacr.org/2023/1568>.
They retain continuous `v=(k-w+ell)/w`; the remaining per-block length rounding,
list-uniformity assumptions, and Gaussian-elimination modeling are not proved
by completing a search.

Equal integer weights are enforced in the list decomposition. For an odd number
of blocks, the two halves use `floor(w/2)` and `ceil(w/2)` separately in success
probabilities, list sizes, and representation counts. This extension replaces
upstream's rounding both halves to the same length. Tests check both scalar
formulas and direct enumeration of small representation sets. The floors of
log2 representation counts use exact rational comparisons near integer boundaries.
Unequal-weight decompositions and additional representation depths are outside
this declared model; this is not a proof they cannot improve an attack.

## Validation and cost

```sh
PYTHONPATH=. python -m unittest discover -s tests -p test_reliable_rsd.py -v
```

The tests compare vectorized search with scalar full-domain enumeration, retain
a large missed-grid regression, check odd block counts, reproduce the controller
bug, and verify failure propagation. Complete ordinary-SD domains can be much
more expensive than the former heuristic windows. A timeout is an unresolved
optimization, not evidence that the current witness is the optimum.
