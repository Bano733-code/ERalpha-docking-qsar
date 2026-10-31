"""Step 7 - Collect all numbers into results/RESULTS.md (paste into the README).

Usage:  python scripts/07_make_report.py
"""
from common import RES, load_json


def g(d, k, f="{:.2f}"):
    v = d.get(k)
    return f.format(v) if isinstance(v, (int, float)) else "n/a"


def main():
    redock = load_json(RES / "redocking.json")
    data = load_json(RES / "data_summary.json")
    qsar = load_json(RES / "qsar_metrics.json")
    scr = load_json(RES / "screening_stats.json")
    lines = [
        "## Results (auto-generated)", "",
        "| Step | Metric | Value |", "|---|---|---|",
        f"| Re-docking 4-OHT | Top-pose RMSD (A) | {g(redock, 'top_pose_rmsd_A')} |",
        f"| Re-docking 4-OHT | Best RMSD among poses (A) | {g(redock, 'best_rmsd_among_poses_A')} |",
        f"| Re-docking 4-OHT | Vina score (kcal/mol) | {g(redock, 'top_pose_score_kcal_mol')} |",
        f"| ChEMBL data | Unique compounds (IC50, nM) | {data['unique_compounds']} |",
        f"| ChEMBL data | Active / inactive (pIC50 >= 6) | {data['n_active']} / {data['n_inactive']} |",
        f"| QSAR (scaffold split) | Test R2 | {g(qsar, 'test_R2')} |",
        f"| QSAR (scaffold split) | Test RMSE (pIC50 units) | {g(qsar, 'test_RMSE')} |",
        f"| QSAR | 5-fold CV R2 (mean +/- sd) | {g(qsar, 'cv5_R2_mean')} +/- {g(qsar, 'cv5_R2_std')} |",
        f"| QSAR | Y-randomisation R2 (mean) | {g(qsar, 'y_randomization_R2_mean')} |",
        f"| QSAR | Test compounds inside AD (Tanimoto >= 0.4) | {g(qsar, 'fraction_test_in_AD', '{:.0%}')} |",
        f"| Virtual screening | Compounds docked | {scr['n_docked']} |",
        f"| Virtual screening | Spearman, -Vina vs pIC50 | {g(scr, 'spearman_docking_vs_pIC50')} |",
        f"| Virtual screening | Spearman, QSAR vs pIC50 | {g(scr, 'spearman_QSAR_vs_pIC50')} |",
        f"| Virtual screening | ROC-AUC docking (actives vs inactives) | {g(scr, 'ROC_AUC_docking_actives_vs_inactives')} |",
        f"| Virtual screening | ROC-AUC QSAR (actives vs inactives) | {g(scr, 'ROC_AUC_QSAR_actives_vs_inactives')} |",
        "",
    ]
    (RES / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
