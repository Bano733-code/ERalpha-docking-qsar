"""Shared helpers for the ERalpha docking + QSAR pipeline.

Every script imports from here, so paths and the docking/ligand preparation
logic live in one place.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
for _d in (RAW, PROC, RES, FIG):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Settings used across the project -------------------------------------
PDB_ID = "3ERT"            # ER-alpha ligand-binding domain + 4-hydroxytamoxifen
CHAIN = "A"
REF_LIGAND = "OHT"         # residue name of 4-hydroxytamoxifen in 3ERT
OHT_SMILES = "CCC(=C(c1ccc(O)cc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"  # no stereo (template only)
CHEMBL_TARGET = "CHEMBL206"  # Estrogen receptor alpha
ACTIVE_PIC50 = 6.0         # pIC50 >= 6  (IC50 <= 1 uM) -> "active"
BOX_SIZE = 22.0            # Angstrom, cubic docking box
SEED = 42


def vina_exe() -> str:
    """Find the AutoDock Vina binary (system PATH first, then ./bin/vina)."""
    found = shutil.which("vina")
    if found:
        return found
    local = ROOT / "bin" / "vina"
    if local.exists():
        return str(local)
    raise FileNotFoundError(
        "AutoDock Vina not found. Run scripts/00_setup.sh (or see README)."
    )


# ---- Receptor / ligand preparation ----------------------------------------
def prepare_receptor(pdb_h_in, pdbqt_out):
    """Protonated receptor PDB -> rigid PDBQT for Vina (Gasteiger charges, AD4 atom types).

    The input must already contain hydrogens (see add_hydrogens_pdbfixer), so Open Babel
    only converts the format and does not add or move any atoms.
    """
    from openbabel import openbabel as ob
    from openbabel import pybel

    mol = next(pybel.readfile("pdb", str(pdb_h_in)))
    ob.OBChargeModel.FindType("gasteiger").ComputeCharges(mol.OBMol)
    mol.write("pdbqt", str(pdbqt_out), opt={"r": None}, overwrite=True)


def add_hydrogens_pdbfixer(pdb_in, pdb_out, ph=7.4):
    """Protonated protein with original residue names/numbers kept (for interaction analysis).

    Open Babel scrambles residue numbering when adding hydrogens, so PDBFixer is used here.
    Missing loops are NOT rebuilt; only missing side-chain atoms and hydrogens are added.
    """
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_in))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(pdb_out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


def embed_ligand(smiles, seed=SEED):
    """SMILES -> RDKit mol with hydrogens and an MMFF-optimised 3D conformer."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("bad SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("3D embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass
    return mol


def mol_to_pdbqt(mol) -> str:
    """RDKit mol (with H, 3D) -> PDBQT string using Meeko."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation().prepare(mol)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"Meeko failed: {err}")
    return pdbqt


def run_vina(receptor, ligand_pdbqt, center, out_pdbqt, size=BOX_SIZE,
             exhaustiveness=8, num_modes=9, seed=SEED, cpu=1):
    """Run AutoDock Vina; return the list of pose scores (kcal/mol, best first)."""
    cmd = [
        vina_exe(), "--receptor", str(receptor), "--ligand", str(ligand_pdbqt),
        "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", str(size), "--size_y", str(size), "--size_z", str(size),
        "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
        "--seed", str(seed), "--cpu", str(cpu), "--out", str(out_pdbqt),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-600:])
    return parse_vina_scores(out_pdbqt)


def parse_vina_scores(pdbqt_path):
    scores = []
    with open(pdbqt_path) as fh:
        for line in fh:
            if line.startswith("REMARK VINA RESULT"):
                scores.append(float(line.split()[3]))
    return scores


def pdbqt_to_rdkit(pdbqt_path):
    """Vina output PDBQT -> RDKit mol; each docked pose is one conformer."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    pm = PDBQTMolecule.from_file(str(pdbqt_path), skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pm)
    return mols[0]


def lipinski_ok(mol):
    """Lipinski rule of five (allows one violation, the usual convention)."""
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    viol = sum([
        Descriptors.MolWt(mol) > 500,
        Crippen.MolLogP(mol) > 5,
        Lipinski.NumHDonors(mol) > 5,
        Lipinski.NumHAcceptors(mol) > 10,
    ])
    return viol <= 1


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
