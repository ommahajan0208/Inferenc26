# Recommended Bid-Ask Table

**Last Updated:** May 9, 2026

**Status:** ✓ Current Best - Degree-4 Legendre Parameterization

## Performance Metrics

- **Objective Score:** 21.96 ± 0.09
- **Margin vs Runner-up:** +0.02 (vs `poly_legendre_deg6` in the same 20×100k run)
- **Testing Methodology:** 20 paired seeds × 100,000 rounds

## Comparison with Closest Runner-up

| Strategy | Objective | Std Dev |
|----------|-----------|---------|
| **poly_legendre_deg4** | 21.96 | 0.09 |
| poly_legendre_deg6 | 21.94 | 0.08 |

## Recommended Table

| Signal | Bid | Ask |
|--------|-----|-----|
| 0 | 6.19 | 90.72 |
| 50 | 6.19 | 137.96 |
| 100 | 15.42 | 190.95 |
| 150 | 47.16 | 246.64 |
| 200 | 88.02 | 300.09 |
| 250 | 134.06 | 350.74 |
| 300 | 183.49 | 399.69 |
| 350 | 234.99 | 447.99 |
| 400 | 287.47 | 496.44 |
| 450 | 340.11 | 545.61 |
| 500 | 392.36 | 595.79 |
| 550 | 443.89 | 647.05 |
| 600 | 494.65 | 699.18 |
| 650 | 544.83 | 751.75 |
| 700 | 594.88 | 804.06 |
| 750 | 645.50 | 855.18 |
| 800 | 697.63 | 903.89 |
| 850 | 752.29 | 948.55 |
| 900 | 808.81 | 985.36 |
| 950 | 862.16 | 1003.96 |
| 1000 | 908.90 | 1003.96 |

## Notes

### Why This Table Wins

The Legendre degree-4 parameterization achieves the best results by:
- **Smoothing the table shape** while preserving useful boundary skew
- Outperforming Legendre degree-6 and the best Chebyshev variant (deg-5) in the latest 20×100k run
- Maintaining stable performance across diverse market conditions

### 🔄 UPDATE PROTOCOL

**This table MUST be updated if any new strategy achieves a higher objective score.**

When updating:
1. Verify the new score with the same testing methodology (≥20 paired seeds, 100,000+ rounds)
2. Include the new performance metrics and comparison table above
3. Document what technique/parameterization achieved the improvement
4. Update the "Last Updated" date
5. Keep the old table entry in version history for reference

## Version History

### May 8, 2026 — poly_chebyshev_deg4 (previous best)

- **Objective Score:** 21.6762 ± 0.0877
- **Testing Methodology:** 20 paired seeds × 100,000 rounds

| Signal | Bid | Ask |
|--------|-----|-----|
| 0 | 5.81 | 94.42 |
| 50 | 5.81 | 140.58 |
| 100 | 5.81 | 192.62 |
| 150 | 32.55 | 247.73 |
| 200 | 72.99 | 300.89 |
| 250 | 120.91 | 351.49 |
| 300 | 173.53 | 400.57 |
| 350 | 228.69 | 449.12 |
| 400 | 284.65 | 497.88 |
| 450 | 340.08 | 547.36 |
| 500 | 394.10 | 597.79 |
| 550 | 446.23 | 649.17 |
| 600 | 496.42 | 701.26 |
| 650 | 545.03 | 753.54 |
| 700 | 592.86 | 805.26 |
| 750 | 641.10 | 855.42 |
| 800 | 691.39 | 902.76 |
| 850 | 745.56 | 945.58 |
| 900 | 803.99 | 979.99 |
| 950 | 862.80 | 997.26 |
| 1000 | 920.50 | 1002.90 |

---

**Validation Status:** ✓ All tests passed
- `python historical_simulator.py --self-test`
- `python -m py_compile` (all modules)
- `python advanced_math_experiments.py --run-next-steps --eval-seeds 20 --eval-n 100000`
