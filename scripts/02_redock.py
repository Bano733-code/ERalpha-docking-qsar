"""Step 2 - Validate the docking set-up by re-docking the co-crystal ligand.

If Vina can put 4-hydroxytamoxifen back where the crystal structure has it
(RMSD < 2 A) then the protein preparation and the docking box are trustworthy.

Usage:  python scripts/02_redock.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from common import FIG, PROC, RES, load_json, mol_to_pdbqt, pdbqt_to_rdkit, run_vina, save_json


def pose_rmsd(pose_mol, ref_mol, conf_id):
    """Symmetry-aware heavy-atom RMSD without re-aligning (docking RMSD)."""
    prb = Chem.RemoveHs(pose_mol)
    ref = Chem.RemoveHs(ref_mol)
    return rdMolAlign.CalcRMS(prb, ref, prbId=conf_id, refId=0)


def plot_overlay(ref_mol, pose_mol, path, rmsd):
    ref = Chem.RemoveHs(ref_mol)
    pose = Chem.RemoveHs(pose_mol)
    a = ref.GetConformer().GetPositions()
    b = pose.GetConformer(0).GetPositions()
    # project onto the 2 main axes of the crystal ligand
    c = a.mean(axis=0)
    _, _, vt = np.linalg.svd(a - c)
    pa, pb = (a - c) @ vt[:2].T, (b - c) @ vt[:2].T
    fig, ax = plt.subplots(figsize=(5, 5))
    for m, p, col, lab in ((ref, pa, "#2a9d8f", "Crystal (3ERT)"),
                           (pose, pb, "#e76f51", "Docked (Vina)")):
        for bond in m.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color=col, lw=2)
        ax.scatter(p[:, 0], p[:, 1], s=14, color=col, label=lab, zorder=3)
    ax.set_aspect("equal")
    ax.set_title(f"Re-docking of 4-hydroxytamoxifen (RMSD = {rmsd:.2f} A)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    box = load_json(PROC / "box.json")
    ref = Chem.SDMolSupplier(str(PROC / "ligand_OHT_crystal.sdf"), removeHs=False)[0]

    lig_pdbqt = PROC / "OHT_input.pdbqt"
    lig_pdbqt.write_text(mol_to_pdbqt(ref))
    out = RES / "OHT_redock.pdbqt"
    scores = run_vina(PROC / "protein.pdbqt", lig_pdbqt, box["center"], out,
                      size=box["size"], exhaustiveness=16, num_modes=9, cpu=2)

    poses = pdbqt_to_rdkit(out)
    rmsds = [pose_rmsd(poses, ref, i) for i in range(poses.GetNumConformers())]
    result = {
        "top_pose_score_kcal_mol": scores[0],
        "top_pose_rmsd_A": rmsds[0],
        "best_rmsd_among_poses_A": min(rmsds),
        "all_scores": scores,
        "all_rmsds": rmsds,
        "validated_under_2A": bool(rmsds[0] < 2.0),
    }
    save_json(result, RES / "redocking.json")
    plot_overlay(ref, poses, FIG / "redocking_overlay.png", rmsds[0])
    print(f"Top pose: {scores[0]:.2f} kcal/mol, RMSD = {rmsds[0]:.2f} A "
          f"(best of all poses: {min(rmsds):.2f} A)")
    print("Docking set-up VALID" if rmsds[0] < 2.0 else
          "Top pose RMSD >= 2 A: check protonation/box, or see best-pose RMSD")


if __name__ == "__main__":
    main()
