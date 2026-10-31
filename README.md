# Integrated Structure-Based and Ligand-Based Pipeline for ERα Inhibitor Screening

A reproducible *in silico* drug-design workflow for the **estrogen receptor alpha (ERα)**, a validated
breast-cancer drug target. It combines **molecular docking** (structure-based) with a **QSAR model**
(ligand-based) and compares both against experimental ChEMBL potencies.

> All numbers below come from running this exact code on PDB 3ERT and 3,390 ChEMBL ERα IC50 compounds
> (`results/RESULTS.md`, `results/*.json`).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/00_setup.sh` | Install packages, AutoDock Vina, download PDB 3ERT |
| 1 | `01_prepare_protein.py` | Clean 3ERT (chain A, no water/hetero atoms), add H at pH 7.4 (PDBFixer), convert to PDBQT (Open Babel, Gasteiger charges), define the docking box around 4-hydroxytamoxifen (OHT) |
| 2 | `02_redock.py` | **Validate docking**: re-dock OHT and compute RMSD to the crystal pose (< 2 Å = valid) |
| 3 | `03_chembl_data.py` | Download ChEMBL ERα (CHEMBL206) IC50 data; keep exact IC50 in nM; remove salts; merge duplicates; pIC50 = 9 − log10(IC50 nM) |
| 4 | `04_qsar.py` | Random Forest on Morgan fingerprints; **scaffold split**, scaffold-grouped 5-fold CV, **Y-randomisation**, applicability domain |
| 5 | `05_screen_dock.py` | Lipinski filter, dock a stratified sample of *hold-out* compounds, compare docking score / QSAR with experiment (Spearman, ROC-AUC) |
| 6 | `06_interactions.py` | ProLIF interaction analysis of the top hits vs. OHT (Ala350, Asp351, Glu353, Leu387, Arg394, His524) |
| 7 | `07_make_report.py` | Collect all metrics into `results/RESULTS.md` |

## How to run

**Google Colab (easiest)** – open `ERalpha_pipeline.ipynb` and run all cells.

**Local (Linux / macOS, Python ≥ 3.10):**
```bash
git clone https://github.com/<your-username>/ERalpha-docking-qsar.git
cd ERalpha-docking-qsar
bash scripts/00_setup.sh      # packages + Vina + 3ERT (3ERT.pdb or 3ERT.cif in data/raw/ both work)
bash scripts/run_all.sh       # runs steps 1-7 (ChEMBL via API)
```
If the ChEMBL API is not reachable, export the CHEMBL206 IC50 activities as CSV from the ChEMBL website,
save it as `data/raw/chembl206_activities.csv` and run `bash scripts/run_all.sh --csv`.
The PDB structure can be given as `data/raw/3ERT.pdb` or `3ERT.cif`.

## Methods in brief

* **Target:** ERα ligand-binding domain, PDB 3ERT (complex with 4-hydroxytamoxifen), chain A.
* **Docking:** AutoDock Vina 1.2.5, cubic box 22 Å centred on the co-crystal ligand, exhaustiveness 8
  (16 for re-docking). Ligands: RDKit ETKDGv3 3D embedding + MMFF94 optimisation, Meeko PDBQT preparation.
* **QSAR:** Morgan fingerprints (radius 2, 2048 bits), Random Forest (500 trees). Scaffold split (80/20) so
  test molecules have unseen Bemis–Murcko scaffolds. Applicability domain: max Tanimoto similarity ≥ 0.4.
* **Fair comparison:** the docked molecules come from the QSAR *test* set, so neither method has seen them.
* **Actives:** pIC50 ≥ 6 (IC50 ≤ 1 µM).

## Results

| Step | Metric | Value |
|---|---|---|
| Re-docking 4-OHT | Top-pose RMSD to crystal pose | **1.23 Å** (valid, < 2 Å); Vina score −9.94 kcal/mol |
| ChEMBL data | Unique compounds after cleaning | 3,390 (2,450 active / 940 inactive, pIC50 ≥ 6) |
| QSAR (scaffold split, 2,712 train / 678 test) | Test R² / RMSE | **0.54 / 1.07** pIC50 units |
| QSAR | 5-fold scaffold-grouped CV R² | 0.55 ± 0.08 |
| QSAR | Y-randomisation R² (10 runs) | −0.38 (model learns real signal) |
| QSAR | Test compounds inside applicability domain | 62 % (MAE 0.66 inside vs 1.04 outside) |
| Virtual screening (100 hold-out compounds) | Spearman ρ, docking score vs pIC50 | 0.48 (p = 4×10⁻⁷) |
| Virtual screening | Spearman ρ, QSAR vs pIC50 | 0.61 (p = 1×10⁻¹¹) |
| Virtual screening | ROC-AUC actives vs inactives: docking / QSAR | 0.76 / 0.88 |

**Interpretation**

* Docking set-up is trustworthy: Vina reproduces the crystal pose of 4-hydroxytamoxifen (RMSD 1.23 Å).
* The scaffold-split QSAR model generalises to new chemotypes (R² 0.54), and Y-randomisation collapses
  performance, so the result is not a fitting artefact.
* Docking scores rank ligands moderately well (ρ = 0.48, AUC 0.76), as expected for Vina on one rigid structure;
  the ligand-based model is stronger here (ρ = 0.61, AUC 0.88) because it learns from thousands of ERα measurements.
  The two methods are complementary: docking gives a binding pose, QSAR gives potency ranking.
* The top-ranked hits (CHEMBL368688, CHEMBL361056, CHEMBL184202, CHEMBL184367, CHEMBL3815076) score −10.5 to −11.4
  kcal/mol, better than re-docked 4-OHT (−9.9). Four of five share a benzoxathiin-type SERM scaffold with a basic
  amine side chain; four of the five contact Leu387 and Arg394 like 4-OHT, and CHEMBL3815076 also contacts Glu353.
  Only 4-OHT made a contact with His524 in this rigid-receptor docking. CHEMBL184367 docked outside the canonical
  pocket (near Cys530/Leu536), so its score should be treated with caution.

| Figure | File |
|---|---|
| Re-docking overlay | `figures/redocking_overlay.png` |
| pIC50 distribution | `figures/pIC50_distribution.png` |
| QSAR performance + applicability domain | `figures/qsar_scatter_AD.png` |
| Most important fingerprint bits | `figures/qsar_top_bits.png` |
| Docking / QSAR vs experiment | `figures/docking_vs_pIC50.png` |
| Interaction heatmap | `figures/interaction_heatmap.png` |

Full machine-generated table: `results/RESULTS.md`. Interactive 3D view of the best complex: `results/top1_complex.html`.

## Limitations

* Vina's scoring function is fast but approximate; weak correlation with experimental potency is expected and
  well documented for diverse ligand sets.
* Rigid receptor, single crystal structure (agonist/antagonist conformation of helix 12 is fixed by 3ERT), no explicit waters, no molecular dynamics. Only 100 of 538 Lipinski-compliant hold-out compounds were docked (stratified by pIC50).
* ChEMBL IC50 values come from different assays and labs (noise of ~0.5 log units is typical).
* The QSAR model is only reliable inside its applicability domain.

## Possible extensions

Ensemble docking against several ERα structures, MM-GBSA rescoring, short MD simulation of the best complexes,
and ADMET/toxicity filters.

## References

* Shiau AK *et al.* (1998) The structural basis of estrogen receptor/coactivator recognition… *Cell* 95:927 (PDB 3ERT).
* Trott O, Olson AJ (2010) AutoDock Vina. *J Comput Chem* 31:455.
* Eberhardt J *et al.* (2021) AutoDock Vina 1.2.0. *J Chem Inf Model* 61:3891.
* Bouysset C, Fiorucci S (2021) ProLIF. *J Cheminform* 13:72.
* Mendez D *et al.* (2019) ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Res* 47:D930.
* RDKit: open-source cheminformatics (rdkit.org).
