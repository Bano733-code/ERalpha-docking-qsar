"""Step 5 - Virtual screening: dock hold-out ER-alpha ligands and compare to experiment.

Compounds come from the QSAR *test* set (never seen by the QSAR model), so docking
and QSAR can be compared fairly against experimental pIC50.
  * Lipinski filter
  * stratified sample over the pIC50 range
  * 3D embedding (RDKit ETKDG + MMFF) -> PDBQT (Meeko) -> AutoDock Vina
  * Spearman correlations and ROC-AUC (active = pIC50 >= 6)

Usage:  python scripts/05_screen_dock.py [--n 100] [--workers 4] [--exhaustiveness 8]
"""
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import (ACTIVE_PIC50, FIG, PROC, RES, SEED, embed_ligand, lipinski_ok,
                    load_json, mol_to_pdbqt, run_vina, save_json)

POSES = RES / "poses_all"


def dock_one(job):
    cid, smiles, box, exh = job
    try:
        mol = embed_ligand(smiles)
        lig = POSES / f"{cid}_in.pdbqt"
        lig.write_text(mol_to_pdbqt(mol))
        out = POSES / f"{cid}.pdbqt"
        score = run_vina(PROC / "protein.pdbqt", lig, box["center"], out,
                         size=box["size"], exhaustiveness=exh, num_modes=5, cpu=1)[0]
        lig.unlink(missing_ok=True)
        return cid, score, None
    except Exception as e:  # keep going if one ligand fails
        return cid, np.nan, str(e)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    args = ap.parse_args()

    POSES.mkdir(parents=True, exist_ok=True)
    box = load_json(PROC / "box.json")
    df = pd.read_csv(RES / "qsar_test_predictions.csv")

    keep = [lipinski_ok(Chem.MolFromSmiles(s)) for s in df["smiles"]]
    n_removed = int(len(df) - sum(keep))
    df = df[keep].reset_index(drop=True)
    print(f"Lipinski filter removed {n_removed} compounds; {len(df)} remain")

    # stratified sample: equal number from each of 5 pIC50 quantile bins
    df["bin"] = pd.qcut(df["pIC50"], 5, labels=False, duplicates="drop")
    per_bin = max(1, args.n // df["bin"].nunique())
    sel = (df.groupby("bin", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), per_bin), random_state=SEED))
             .reset_index(drop=True))
    print(f"Docking {len(sel)} compounds")

    jobs = [(r.chembl_id, r.smiles, box, args.exhaustiveness) for r in sel.itertuples()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(dock_one, jobs))
    sel["vina_score"] = [r[1] for r in results]
    fails = [(r[0], r[2]) for r in results if r[2]]
    if fails:
        print(f"{len(fails)} ligands failed, e.g. {fails[:3]}")
    sel = sel.dropna(subset=["vina_score"]).copy()
    sel.to_csv(RES / "docking_results.csv", index=False)

    # --- statistics (docking score is negative = better, so flip the sign) ---
    d = -sel["vina_score"]
    rho_dock, p_dock = spearmanr(d, sel["pIC50"])
    rho_qsar, p_qsar = spearmanr(sel["qsar_pred_pIC50"], sel["pIC50"])
    rho_dq, _ = spearmanr(d, sel["qsar_pred_pIC50"])
    stats = {
        "n_docked": int(len(sel)), "lipinski_removed": n_removed,
        "spearman_docking_vs_pIC50": float(rho_dock), "p_docking": float(p_dock),
        "spearman_QSAR_vs_pIC50": float(rho_qsar), "p_QSAR": float(p_qsar),
        "spearman_docking_vs_QSAR": float(rho_dq),
    }
    act = (sel["pIC50"] >= ACTIVE_PIC50).astype(int)
    if act.nunique() == 2:
        stats["ROC_AUC_docking_actives_vs_inactives"] = float(roc_auc_score(act, d))
        stats["ROC_AUC_QSAR_actives_vs_inactives"] = float(roc_auc_score(act, sel["qsar_pred_pIC50"]))
    stats["n_active"] = int(act.sum())
    save_json(stats, RES / "screening_stats.json")
    print(stats)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ref = RES / "redocking.json"
    for ax, x, xl, rho in ((axes[0], d, "-Vina score (kcal/mol, higher = better)", rho_dock),
                           (axes[1], sel["qsar_pred_pIC50"], "QSAR predicted pIC50", rho_qsar)):
        ax.scatter(x, sel["pIC50"], s=18, alpha=0.7, color="#2a9d8f")
        ax.set_xlabel(xl)
        ax.set_ylabel("Experimental pIC50")
        ax.set_title(f"Spearman rho = {rho:.2f}")
    if ref.exists():
        axes[0].axvline(-load_json(ref)["top_pose_score_kcal_mol"], color="#e76f51", ls="--",
                        label="4-OHT re-docked")
        axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "docking_vs_pIC50.png", dpi=200)
    plt.close(fig)

    # keep the 5 best-scoring poses for the interaction analysis
    top = sel.nsmallest(5, "vina_score")
    (RES / "top_poses").mkdir(exist_ok=True)
    for r in top.itertuples():
        shutil.copy(POSES / f"{r.chembl_id}.pdbqt", RES / "top_poses" / f"{r.chembl_id}.pdbqt")
    top.to_csv(RES / "top5_hits.csv", index=False)


if __name__ == "__main__":
    main()
