#!/usr/bin/env python
"""Tables 3 and 4 from perceptual-study counts.

Input CSV columns: category,method,correct,total   (method in {despina, baseline})
configs/human_study_paper_counts.csv holds the study's counts (36 judgements per cell):
  flat,despina,8,36 / flat,baseline,23,36 / hilly,despina,12,36 / hilly,baseline,30,36 /
  rocky,despina,10,36 / rocky,baseline,18,36
"""
import argparse, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina.stats import binom_test_vs_chance, two_proportion_z, wilson_ci  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    rows = []
    for cat, g in df.groupby("category", sort=False):
        d = g[g.method == "despina"].iloc[0]; b = g[g.method == "baseline"].iloc[0]
        lo1, hi1 = wilson_ci(d.correct, d.total); lo2, hi2 = wilson_ci(b.correct, b.total)
        rows.append({"category": cat,
                     "despina_acc": round(d.correct / d.total, 2), "despina_ci": f"{lo1:.2f}-{hi1:.2f}",
                     "baseline_acc": round(b.correct / b.total, 2), "baseline_ci": f"{lo2:.2f}-{hi2:.2f}",
                     "p_despina_vs_chance": round(binom_test_vs_chance(d.correct, d.total), 4),
                     "p_baseline_vs_chance": round(binom_test_vs_chance(b.correct, b.total), 4),
                     "p_despina_vs_baseline": round(two_proportion_z(d.correct, d.total, b.correct, b.total), 4)})
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
