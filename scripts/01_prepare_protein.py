"""Step 1 - Prepare the ER-alpha protein (PDB 3ERT).

What this does:
  * keeps chain A, removes water and all hetero atoms (cleaned protein)
  * extracts the co-crystallised ligand 4-hydroxytamoxifen (OHT)
  * adds hydrogens at pH 7.4 and writes the PDBQT file Vina needs
  * defines the docking box around OHT

Usage:  python scripts/01_prepare_protein.py [--pdb data/raw/3ERT.pdb]
"""
import argparse

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from rdkit import Chem
from rdkit.Chem import AllChem

from common import (BOX_SIZE, CHAIN, OHT_SMILES, PROC, RAW, REF_LIGAND,
                    PDB_ID, add_hydrogens_pdbfixer, prepare_receptor, save_json)


class ProteinOnly(Select):
    def accept_chain(self, chain):
        return chain.id == CHAIN

    def accept_residue(self, res):
        return res.id[0] == " "  # standard residues only (drops water + HETATM)


class LigandOnly(Select):
    def accept_chain(self, chain):
        return chain.id == CHAIN

    def accept_residue(self, res):
        return res.get_resname() == REF_LIGAND


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=None, help="3ERT structure (.pdb or .cif)")
    args = ap.parse_args()

    path = args.pdb
    if path is None:
        path = next((str(p) for p in (RAW / f"{PDB_ID}.pdb", RAW / f"{PDB_ID}.cif") if p.exists()), None)
    if path is None:
        raise SystemExit(f"Put {PDB_ID}.pdb or {PDB_ID}.cif into data/raw/")
    parser = MMCIFParser(QUIET=True) if path.lower().endswith(".cif") else PDBParser(QUIET=True)
    structure = parser.get_structure(PDB_ID, path)
    io = PDBIO()
    io.set_structure(structure)

    prot_pdb = PROC / "protein_A.pdb"
    lig_pdb = PROC / "ligand_OHT_crystal.pdb"
    io.save(str(prot_pdb), ProteinOnly())
    io.save(str(lig_pdb), LigandOnly())

    # Box centre = mean position of the OHT heavy atoms
    coords = np.array([a.coord for a in structure[0][CHAIN].get_atoms()
                       if a.get_parent().get_resname() == REF_LIGAND])
    if len(coords) == 0:
        raise SystemExit(f"Ligand {REF_LIGAND} not found in chain {CHAIN}.")
    center = coords.mean(axis=0).tolist()

    # Give the crystal ligand correct bond orders using its SMILES as template
    crystal = Chem.MolFromPDBFile(str(lig_pdb), removeHs=True)
    template = Chem.MolFromSmiles(OHT_SMILES)
    ref = AllChem.AssignBondOrdersFromTemplate(template, crystal)
    ref = Chem.AddHs(ref, addCoords=True)
    w = Chem.SDWriter(str(PROC / "ligand_OHT_crystal.sdf"))
    w.write(ref)
    w.close()

    # 1) add hydrogens at pH 7.4 with PDBFixer (keeps residue names/numbers)
    add_hydrogens_pdbfixer(prot_pdb, PROC / "protein_H.pdb")
    # 2) convert to PDBQT for Vina; the same protonated file is used for ProLIF later
    prepare_receptor(PROC / "protein_H.pdb", PROC / "protein.pdbqt")

    save_json({"center": center, "size": BOX_SIZE, "n_ligand_atoms": int(len(coords))},
              PROC / "box.json")
    print(f"Protein prepared. Box centre = {np.round(center, 2)}, size = {BOX_SIZE} A")


if __name__ == "__main__":
    main()
