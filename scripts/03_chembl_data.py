"""Step 3 - Get and clean ER-alpha (CHEMBL206) IC50 data from ChEMBL.

Two ways to provide the data:
  1) automatic (needs internet):   python scripts/03_chembl_data.py --source api
  2) from a CSV you downloaded from the ChEMBL website:
        python scripts/03_chembl_data.py --source csv --csv data/raw/chembl206_activities.csv

Cleaning: IC50 only, nM units, exact ('=') values, salts removed, canonical SMILES,
duplicates merged (median), pIC50 = 9 - log10(IC50 in nM).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover  # noqa: F401  (kept for clarity)
from rdkit.Chem.MolStandardize import rdMolStandardize

from common import ACTIVE_PIC50, CHEMBL_TARGET, FIG, PROC, RAW, RES, save_json


def load_from_api():
    from chembl_webresource_client.new_client import new_client
    acts = new_client.activity.filter(
        target_chembl_id=CHEMBL_TARGET, standard_type="IC50",
        standard_units="nM", standard_relation="=",
    ).only(["molecule_chembl_id", "canonical_smiles", "standard_type",
            "standard_value", "standard_units", "standard_relation"])
    return pd.DataFrame(list(acts))


def load_from_csv(path):
    df = pd.read_csv(path, sep=";")  # ChEMBL website exports are ';'-separated
    if df.shape[1] < 3:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {"smiles": "canonical_smiles", "molecule_chembl_id": "molecule_chembl_id"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    needed = {"molecule_chembl_id", "canonical_smiles", "standard_type",
              "standard_value", "standard_units", "standard_relation"}
    for c in ("standard_relation", "standard_type", "standard_units"):
        df[c] = df[c].astype(str).str.strip().str.strip("'\"")  # ChEMBL export writes '=' with quotes
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    return df


def clean_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)  # drop salts/counter-ions
    return Chem.MolToSmiles(mol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["api", "csv"], default="api")
    ap.add_argument("--csv", default=str(RAW / "chembl206_activities.csv"))
    args = ap.parse_args()

    df = load_from_api() if args.source == "api" else load_from_csv(args.csv)
    n_raw = len(df)
    # website exports write the relation as '=' (with quotes); normalise
    df["standard_relation"] = df["standard_relation"].astype(str).str.strip().str.strip("'\"")

    df = df[(df["standard_type"] == "IC50") & (df["standard_units"] == "nM")
            & (df["standard_relation"] == "=")].copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["standard_value"] > 0]
    n_filtered = len(df)

    df["smiles"] = df["canonical_smiles"].map(clean_smiles)
    df = df.dropna(subset=["smiles"])

    # one row per molecule: median IC50 over duplicate measurements
    g = df.groupby("smiles").agg(
        chembl_id=("molecule_chembl_id", "first"),
        ic50_nM=("standard_value", "median"),
        n_measurements=("standard_value", "size"),
    ).reset_index()
    g["pIC50"] = 9 - np.log10(g["ic50_nM"])
    g["active"] = (g["pIC50"] >= ACTIVE_PIC50).astype(int)
    g = g[(g["pIC50"] > 0) & (g["pIC50"] < 13)]  # drop physically absurd values
    g.to_csv(PROC / "chembl_clean.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(g["pIC50"], bins=30, color="#2a9d8f", edgecolor="white")
    ax.axvline(ACTIVE_PIC50, color="#e76f51", ls="--", label=f"active threshold (pIC50 = {ACTIVE_PIC50:g})")
    ax.set_xlabel("pIC50")
    ax.set_ylabel("Number of compounds")
    ax.set_title(f"ER-alpha ({CHEMBL_TARGET}) IC50 data, n = {len(g)}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "pIC50_distribution.png", dpi=200)

    summary = {"raw_rows": n_raw, "after_filters": n_filtered, "unique_compounds": len(g),
               "n_active": int(g["active"].sum()), "n_inactive": int((1 - g["active"]).sum()),
               "pIC50_min": float(g["pIC50"].min()), "pIC50_max": float(g["pIC50"].max())}
    save_json(summary, RES / "data_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
