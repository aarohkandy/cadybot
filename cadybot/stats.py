"""Small-sample statistics. Pure functions: no database, no config, no model.

snapshot.py is the module that guarantees cadybot cannot hallucinate a number.
This module is the one that guarantees the numbers it does print are not
themselves fiction — a percentage over seven people looks exactly as precise as
a percentage over seven thousand, and at seven it means nothing.

The floors below are duplicated in config.py, where the rest of the codebase
reads them. They are restated here rather than imported because a statistics
module that imports application configuration stops being testable in
isolation; snapshot.py asserts the two copies agree at import time, so they
cannot drift.
"""

import math
from typing import Dict, List, Optional, Tuple

# Events needed before a percentage is printed at all. Below this a rate is
# reported as a bare count over a bare denominator, which is the whole truth.
MIN_RATE_DENOMINATOR = 20
# Distinct posters and directed edges needed before a reply graph means
# anything. Under these the null sampling band on rho is about +/-0.40.
MIN_RECIPROCITY_POSTERS = 15
MIN_RECIPROCITY_EDGES = 30


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    The normal approximation collapses at small n and at p near 0 or 1 — it
    happily returns a negative lower bound — which is exactly the regime
    cadybot lives in.

    >>> [round(x, 2) for x in wilson(3, 7)]
    [0.16, 0.75]
    """
    if n <= 0:
        return (0.0, 1.0)
    p = float(k) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_of_three(n: int) -> float:
    """95% upper bound on a rate whose observed count is zero.

    Seeing no leaves in 30 members does not mean the leave rate is 0; it means
    it is under 3/30 = 10%. Returns 1.0 for n == 0, where nothing is bounded.
    """
    if n <= 0:
        return 1.0
    return min(1.0, 3.0 / n)


def render_rate(k: int, n: int) -> Dict[str, object]:
    """The single gate every member-denominated rate in the snapshot passes.

    Below MIN_RATE_DENOMINATOR there is no `pct` key at all. Present-but-
    caveated does not work: the model quotes the number and drops the caveat,
    and so does the founder reading the model. At n=7 the 95% Wilson half-width
    at p=0.5 is +/-29.6pp, so "43% never posted" is really somewhere in
    [16%, 75%] — a statement with no content.

    >>> render_rate(3, 7) == {"count": 3, "of": 7}
    True
    >>> sorted(render_rate(30, 100))
    ['ci_high', 'ci_low', 'count', 'of', 'pct']
    """
    out = {"count": k, "of": n}  # type: Dict[str, object]
    if n < MIN_RATE_DENOMINATOR:
        return out
    low, high = wilson(k, n)
    out["pct"] = round(100.0 * k / n)
    out["ci_low"] = round(100.0 * low, 1)
    out["ci_high"] = round(100.0 * high, 1)
    return out


def count_change_pvalue(x1: int, x2: int, p0: float = 0.5) -> float:
    """Przyborowski-Wilenski conditional binomial C-test on two Poisson counts.

    "Messages went from 40 to 55 this week" is the sentence cadybot is most
    tempted to build advice on, and it is usually noise. Conditional on the
    total k = x1 + x2, X1 is Binomial(k, p0) with p0 = t1 / (t1 + t2) for
    exposures t1 and t2; the two-sided exact p is the total probability of every
    outcome no more likely than the one observed.

    Always pass p0 explicitly as t1 / (t1 + t2). Comparing a five-day window to
    a seven-day one at the default p0 = 0.5 manufactures a result.

    The tail carries only half the mass of the outcomes exactly as likely as the
    observed one. That is Lancaster's mid-p: on a discrete statistic the strict
    exact test is conservative enough that its actual size sits well under the
    nominal 5%, and here that conservatism points the wrong way — it would let
    cadybot call a real change unremarkable. The two anchors below are the
    published worked values for this test and the strict form misses both
    (0.151 and 0.057).

    >>> round(count_change_pvalue(40, 55), 2)
    0.13
    >>> round(count_change_pvalue(40, 60), 3)
    0.046
    >>> count_change_pvalue(0, 0)
    1.0
    """
    k = x1 + x2
    if k <= 0:
        return 1.0
    q0 = 1.0 - p0
    pmf = [math.comb(k, i) * (p0 ** i) * (q0 ** (k - i)) for i in range(k + 1)]
    # The slack is what makes the mirror-image outcome of a symmetric test count
    # as tied rather than as strictly smaller; floating-point drift alone would
    # otherwise put it on one side or the other at random.
    ceiling = pmf[x1] * (1.0 + 1e-9)
    floor = pmf[x1] * (1.0 - 1e-9)
    smaller = math.fsum(p for p in pmf if p < floor)
    tied = math.fsum(p for p in pmf if floor <= p <= ceiling)
    return min(1.0, smaller + 0.5 * tied)


def bus_factor(counts: List[int]) -> int:
    """CHAOSS Contributor Absence Factor: how few people produce half the volume.

    An integer in [1, n] at every server size. It imports no false precision —
    "2 of 8 contributors produce half the messages" is either true or it is not
    — which is why it stands in for Gini, whose value at seed stage is dominated
    by how many silent members happen to be counted in the denominator.

    >>> bus_factor([1000, 202, 90, 33, 332, 343, 42, 433])
    2
    >>> bus_factor([])
    0
    """
    total = sum(counts)
    if total <= 0:
        return 0
    threshold = total / 2.0
    running = 0
    for i, c in enumerate(sorted(counts, reverse=True)):
        running += c
        if running >= threshold:
            return i + 1
    return len(counts)


def reciprocity(
    edges: Dict[Tuple[int, int], int], n_posters: int
) -> Optional[float]:
    """Garlaschelli & Loffredo density-corrected reciprocity, or None.

    rho = (r - a_bar) / (1 - a_bar), where r is the fraction of directed edges
    whose reverse also exists and a_bar = L / (n * (n - 1)) is the graph
    density. The uncorrected r reads a seven-person server as intensely
    reciprocal purely because seven people in one channel are nearly fully
    connected; the correction subtracts exactly that.

    Returns None — never 0.0 — below the gates. A missing value the model must
    refuse to interpret beats a number whose null band is +/-0.40.
    """
    if n_posters < MIN_RECIPROCITY_POSTERS or n_posters < 2:
        return None
    directed = [pair for pair, weight in edges.items() if weight > 0 and pair[0] != pair[1]]
    total_edges = len(directed)
    if total_edges < MIN_RECIPROCITY_EDGES:
        return None
    present = set(directed)
    r = sum(1 for (a, b) in directed if (b, a) in present) / float(total_edges)
    a_bar = total_edges / float(n_posters * (n_posters - 1))
    if a_bar >= 1.0:
        return None  # complete graph: reciprocity is undefined, not perfect
    return round((r - a_bar) / (1.0 - a_bar), 3)


def median(values: List[float]) -> Optional[float]:
    """Median, or None for an empty sample. Never raises on no data."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
