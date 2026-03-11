"""
src/utils.py
Utility functions for RNA 3D structure prediction.

Covers:
  - Sequence parsing (FASTA / raw string)
  - PDB loading and coordinate extraction
  - RMSD calculation
  - Coordinate formatting for submission
  - Random seed setting for reproducibility
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────

# RNA nucleotide vocabulary
RNA_VOCAB: Dict[str, int] = {"A": 0, "U": 1, "G": 2, "C": 3, "<PAD>": 4, "<UNK>": 5}
RNA_VOCAB_INV: Dict[int, str] = {v: k for k, v in RNA_VOCAB.items()}

# Atom names used for C1' (backbone) and base atoms
BACKBONE_ATOMS = ["C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O5'", "P"]
BASE_HEAVY_ATOMS = ["N1", "N3", "N9", "C2", "C4", "C6", "C8"]


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Makes CUDA deterministic (slight perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed] All random seeds set to {seed}")


# ── Device Helpers ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        print(f"[device] Using CUDA: {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("[device] Using CPU")
    return device


# ── Sequence Parsing ───────────────────────────────────────────────────────────

def parse_fasta(path: str | Path) -> Dict[str, str]:
    """
    Parse a FASTA file into {identifier: sequence} dict.
    Handles multi-line sequences.
    """
    records: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_seq: List[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper().replace("T", "U"))  # DNA→RNA

    if current_id is not None:
        records[current_id] = "".join(current_seq)

    return records


def encode_sequence(seq: str) -> torch.Tensor:
    """
    One-hot encode an RNA sequence string.
    Returns shape (L, 4) — A, U, G, C.
    Unknown nucleotides map to all-zeros.
    """
    L = len(seq)
    enc = torch.zeros(L, 4, dtype=torch.float32)
    for i, nt in enumerate(seq):
        idx = RNA_VOCAB.get(nt, -1)
        if 0 <= idx < 4:
            enc[i, idx] = 1.0
    return enc


def sequence_to_indices(seq: str) -> torch.Tensor:
    """Convert RNA string to integer index tensor (L,)."""
    return torch.tensor(
        [RNA_VOCAB.get(nt, RNA_VOCAB["<UNK>"]) for nt in seq],
        dtype=torch.long,
    )


# ── PDB Parsing ────────────────────────────────────────────────────────────────

def load_pdb_coords(
    pdb_path: str | Path,
    atom_name: str = "C3'",
    chain_id: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract Cα-equivalent (C3') coordinates from a PDB file.

    Args:
        pdb_path:  Path to .pdb file.
        atom_name: Atom to extract. Default C3' (RNA backbone proxy).
        chain_id:  If given, only extract from this chain.

    Returns:
        coords:   (N, 3) float32 numpy array of xyz.
        residues: List of residue names.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna", str(pdb_path))

    coords: List[np.ndarray] = []
    residues: List[str] = []

    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] != " ":  # skip HETATM / water
                    continue
                if atom_name in residue:
                    coords.append(residue[atom_name].get_vector().get_array())
                    residues.append(residue.get_resname().strip())
        break  # first model only

    if not coords:
        warnings.warn(f"No {atom_name} atoms found in {pdb_path}")
        return np.zeros((0, 3), dtype=np.float32), []

    return np.array(coords, dtype=np.float32), residues


def load_all_heavy_atoms(pdb_path: str | Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load ALL heavy atom coordinates for MDAnalysis-style analysis.

    Returns:
        coords:      (N_atoms, 3)
        atom_names:  List of atom names
        res_names:   List of residue names per atom
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna", str(pdb_path))

    coords, atom_names, res_names = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                for atom in residue:
                    if atom.element != "H":  # skip hydrogens
                        coords.append(atom.get_vector().get_array())
                        atom_names.append(atom.get_name())
                        res_names.append(residue.get_resname().strip())
        break
    return np.array(coords, dtype=np.float32), atom_names, res_names


# ── RMSD ───────────────────────────────────────────────────────────────────────

def rmsd(pred: np.ndarray, true: np.ndarray, aligned: bool = True) -> float:
    """
    Compute RMSD between predicted and true coordinate sets.
    Optionally performs Kabsch alignment first (default: True).

    Args:
        pred:    (N, 3) predicted coordinates.
        true:    (N, 3) ground-truth coordinates.
        aligned: If True, apply Kabsch rotation before RMSD.

    Returns:
        RMSD value in Ångströms.
    """
    assert pred.shape == true.shape, "Shape mismatch for RMSD"
    if aligned:
        pred = kabsch_align(pred, true)
    diff = pred - true
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Kabsch algorithm: finds optimal rotation to align mobile → target.
    Both arrays are (N, 3).
    """
    mobile = mobile - mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    target_c = target - target_center

    H = mobile.T @ target_c
    U, _, Vt = np.linalg.svd(H)
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    return (mobile @ R.T) + target_center


def tm_score_approx(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Approximate TM-score (full version requires TMalign binary).
    Uses the L_norm d0 formula.
    L ≥ 22 required for meaningful scores.
    """
    L = len(true)
    if L < 22:
        return float("nan")
    d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8
    d0 = max(d0, 0.5)
    aligned = kabsch_align(pred, true)
    d = np.sqrt(((aligned - true) ** 2).sum(axis=1))
    return float((1 / (1 + (d / d0) ** 2)).mean())


# ── Coordinate Formatting ─────────────────────────────────────────────────────

def coords_to_pdb_string(
    coords: np.ndarray,
    sequence: str,
    chain_id: str = "A",
    model_num: int = 1,
) -> str:
    """
    Convert (L, 3) coordinate array to minimal PDB-format string.
    One C3' atom per nucleotide.

    Args:
        coords:    (L, 3) coordinates in Ångströms.
        sequence:  RNA nucleotide string of length L.
        chain_id:  Chain identifier character.
        model_num: MODEL record number.

    Returns:
        PDB-format string.
    """
    lines = [f"MODEL     {model_num:4d}"]
    atom_idx = 1
    for i, (nt, xyz) in enumerate(zip(sequence, coords)):
        x, y, z = xyz
        lines.append(
            f"ATOM  {atom_idx:5d}  C3' {nt:3s} {chain_id}{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        atom_idx += 1
    lines.append("ENDMDL")
    return "\n".join(lines) + "\n"


def save_pdb(
    coords_list: List[np.ndarray],
    sequence: str,
    out_path: str | Path,
    chain_id: str = "A",
) -> None:
    """
    Save multiple predicted structures as MODEL records in a single PDB file.
    Competition requires 5 structures per sequence.

    Args:
        coords_list: List of (L, 3) arrays (one per model).
        sequence:    RNA sequence string.
        out_path:    Output .pdb file path.
        chain_id:    Chain identifier.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, coords in enumerate(coords_list, start=1):
            f.write(coords_to_pdb_string(coords, sequence, chain_id, model_num=i))
    print(f"[save] Saved {len(coords_list)} model(s) to {out_path}")
