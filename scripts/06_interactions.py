"""Step 6 - Protein-ligand interaction analysis of the top docking hits.

Compares the interactions of the top-5 hits with those of the re-docked
4-hydroxytamoxifen. ER-alpha key residues: Glu353, Arg394, Leu387, His524
(plus Asp351 and Ala350 for tamoxifen-like ligands).

Output: results/interactions_top_hits.csv, figures/interaction_heatmap.png,
        results/top1_complex.html (interactive 3D view)

Usage:  python scripts/06_interactions.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import prolif as plf
import seaborn as sns
from rdkit import Chem

from common import FIG, PROC, RES, pdbqt_to_rdkit

KEY = ["ALA350", "ASP351", "GLU353", "LEU387", "ARG394", "HIS524"]
INTERACTIONS = ["HBDonor", "HBAcceptor", "Hydrophobic", "PiStacking",
                "Anionic", "Cationic", "CationPi", "PiCation"]


def best_pose(pdbqt_path):
    mol = pdbqt_to_rdkit(pdbqt_path)
    one = Chem.Mol(mol)
    one.RemoveAllConformers()
    one.AddConformer(Chem.Conformer(mol.GetConformer(0)), assignId=True)
    return one


def main():
    items = {"4-OHT (re-docked)": RES / "OHT_redock.pdbqt"}
    for p in sorted((RES / "top_poses").glob("*.pdbqt")):
        items[p.stem] = p

    ligs = {name: best_pose(p) for name, p in items.items()}
    # Protein: read the heavy-atom PDB with RDKit (standard residue templates give correct
    # bonds), then add hydrogens. This is more robust than guessing bonds from the
    # Open Babel hydrogen file.
    prot_rd = Chem.MolFromPDBFile(str(PROC / "protein_A.pdb"), removeHs=False)
    prot_rd = Chem.AddHs(prot_rd, addCoords=True, addResidueInfo=True)
    prot = plf.Molecule.from_rdkit(prot_rd)
    fp = plf.Fingerprint(INTERACTIONS)
    fp.run_from_iterable([plf.Molecule.from_rdkit(m) for m in ligs.values()], prot, progress=False)
    df = fp.to_dataframe()
    df.index = list(ligs.keys())

    # tidy table: ligand | residue | interaction
    rows = []
    for name in df.index:
        for (_, res, inter), val in df.loc[name].items():
            if val:
                rows.append({"ligand": name, "residue": str(res).split(".")[0], "interaction": inter})
    tidy = pd.DataFrame(rows)
    tidy.to_csv(RES / "interactions_top_hits.csv", index=False)

    if tidy.empty:
        print("No interactions detected - check the poses.")
        return
    mat = tidy.groupby(["ligand", "residue"]).size().unstack(fill_value=0)
    # keep residues that are key or contacted by at least 2 ligands
    cols = [c for c in mat.columns if c in KEY or (mat[c] > 0).sum() >= 2]
    mat = mat[cols].reindex(columns=sorted(cols, key=lambda r: int("".join(ch for ch in r if ch.isdigit()))))
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(cols) + 3), 3.5))
    sns.heatmap(mat, annot=True, cmap="Greens", cbar_kws={"label": "# interactions"}, ax=ax)
    ax.set_title("ER-alpha residues contacted by re-docked 4-OHT and the top hits")
    fig.tight_layout()
    fig.savefig(FIG / "interaction_heatmap.png", dpi=200)
    plt.close(fig)

    present = {k: [n for n in mat.index if k in mat.columns and mat.loc[n, k] > 0] for k in KEY}
    print("Key residue contacts:", present)

    # interactive 3D view of the best hit
    try:
        import py3Dmol
        top1 = [n for n in ligs if n != "4-OHT (re-docked)"][0]
        view = py3Dmol.view(width=800, height=600)
        view.addModel(open(PROC / "protein_H.pdb").read(), "pdb")
        view.setStyle({"model": 0}, {"cartoon": {"color": "lightgray"}})
        view.addModel(Chem.MolToMolBlock(ligs[top1]), "mol")
        view.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon"}})
        view.addModel(Chem.MolToMolBlock(ligs["4-OHT (re-docked)"]), "mol")
        view.setStyle({"model": 2}, {"stick": {"colorscheme": "orangeCarbon"}})
        view.zoomTo({"model": 1})
        view.write_html(str(RES / "top1_complex.html"))
    except Exception as e:
        print("3D view skipped:", e)


if __name__ == "__main__":
    main()
