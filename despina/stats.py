"""Statistics for the perceptual study (Tables 3 and 4)."""
from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def binom_test_vs_chance(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test against chance. This is the test that
    reproduces the 'vs. Chance' row of Table 4 (with 36 judgements per cell)."""
    from scipy.stats import binomtest
    return float(binomtest(k, n, p0).pvalue) if n else float("nan")


def z_test_vs_chance(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided one-proportion z-test against chance."""
    if n == 0:
        return float("nan")
    z = (k / n - p0) / math.sqrt(p0 * (1 - p0) / n)
    return 2 * _norm_sf(abs(z))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided pooled two-proportion z-test."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return 2 * _norm_sf(abs(z))
