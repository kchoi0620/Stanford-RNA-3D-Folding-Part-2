"""
src/data_io.py
Data loading utilities for Stanford RNA 3D Folding — Part 2.

Handles:
  - Loading competition CSV files (train_sequences.csv, etc.)
  - Building PyTorch Geometric graphs from RNA sequences
  - Dataset and DataLoader wrappers
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData

from .utils import encode_sequence, load_pdb_coords, sequence_to_indices

# ── Competition File Paths ─────────────────────────────────────────────────────

DATA_DIR = Path("./data")
TRAIN_CSV = DATA_DIR / "train_sequences.csv"
TEST_CSV = DATA_DIR / "test_sequences.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
PDB_DIR = DATA_DIR / "pdbs"
OUTPUT_DIR = Path("./output")
SUBMISSION_DIR = OUTPUT_DIR / "submission"


def load_train_df(path: Path = TRAIN_CSV) -> pd.DataFrame:
    """
    Load the training sequences CSV.

    Expected columns: sequence_id, sequence, (optional) structure, length.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {path}. "
            "Download competition data with: kaggle competitions download "
            "-c stanford-rna-3d-folding-part2"
        )
    df = pd.read_csv(path)
    if "length" not in df.columns and "sequence" in df.columns:
        df["length"] = df["sequence"].str.len()
    print(f"[data] Loaded {len(df):,} training sequences  |  columns: {list(df.columns)}")
    return df


def load_test_df(path: Path = TEST_CSV) -> pd.DataFrame:
    """Load test sequences CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found at {path}")
    df = pd.read_csv(path)
    if "length" not in df.columns and "sequence" in df.columns:
        df["length"] = df["sequence"].str.len()
    print(f"[data] Loaded {len(df):,} test sequences")
    return df


# ── RNA → Graph ────────────────────────────────────────────────────────────────

def rna_to_graph(
    sequence: str,
    coords: Optional[np.ndarray] = None,
    include_long_range: bool = False,
    long_range_cutoff: float = 8.0,
) -> PyGData:
    """
    Convert an RNA sequence to a PyTorch Geometric graph.

    Graph construction:
      - Nodes    = nucleotides (one per residue)
      - Node features = 4-dim one-hot (A/U/G/C) + position encoding
      - Edges    = backbone bonds (i→i+1) + optional spatial contacts

    Args:
        sequence:            RNA nucleotide string.
        coords:              (L, 3) C3' coordinates. If None, no spatial edges.
        include_long_range:  Add edges between residues within cutoff distance.
        long_range_cutoff:   Distance threshold in Å for spatial edges.

    Returns:
        PyTorch Geometric Data object.
    """
    L = len(sequence)

    # Node features: one-hot (4) + sinusoidal position embedding (16)
    one_hot = encode_sequence(sequence)  # (L, 4)
    pos_enc = _sinusoidal_position_encoding(L, dim=16)  # (L, 16)
    x = torch.cat([one_hot, pos_enc], dim=-1)  # (L, 20)

    # Backbone edges: i → i+1 (bidirectional)
    src = list(range(L - 1)) + list(range(1, L))
    dst = list(range(1, L)) + list(range(L - 1))
    edge_types = [0] * len(src)  # 0 = backbone

    # Spatial edges from coordinates
    if coords is not None and include_long_range:
        dist_mat = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :], axis=-1
        )  # (L, L)
        i_idx, j_idx = np.where(
            (dist_mat < long_range_cutoff) & (np.abs(
                np.arange(L)[:, None] - np.arange(L)[None, :]
            ) > 1)
        )
        src += i_idx.tolist()
        dst += j_idx.tolist()
        edge_types += [1] * len(i_idx)  # 1 = spatial contact

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_types, dtype=torch.long).unsqueeze(-1)

    # Node-level target: 3D coordinates (L, 3) if available
    y = torch.tensor(coords, dtype=torch.float32) if coords is not None else None

    return PyGData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr.float(),
        y=y,
        seq=sequence,
        num_nodes=L,
    )


def _sinusoidal_position_encoding(length: int, dim: int = 16) -> torch.Tensor:
    """Standard sinusoidal position encoding. Returns (length, dim)."""
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * (-np.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA 3D structure prediction.

    Yields PyTorch Geometric Data objects.
    Set `pdb_dir=None` for test set (no ground-truth coordinates).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pdb_dir: Optional[Path] = None,
        seq_col: str = "sequence",
        id_col: str = "sequence_id",
        max_len: Optional[int] = None,
    ):
        self.df = df.copy()
        if max_len is not None:
            self.df = self.df[self.df["length"] <= max_len].reset_index(drop=True)
            print(f"[dataset] Filtered to {len(self.df):,} sequences with len ≤ {max_len}")

        self.pdb_dir = pdb_dir
        self.seq_col = seq_col
        self.id_col = id_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> PyGData:
        row = self.df.iloc[idx]
        seq_id = str(row[self.id_col])
        sequence = str(row[self.seq_col]).upper().replace("T", "U")

        coords = None
        if self.pdb_dir is not None:
            pdb_path = self.pdb_dir / f"{seq_id}.pdb"
            if pdb_path.exists():
                try:
                    coords, _ = load_pdb_coords(pdb_path)
                    if len(coords) != len(sequence):
                        coords = None  # length mismatch — skip coords
                except Exception:
                    coords = None

        graph = rna_to_graph(sequence, coords=coords, include_long_range=coords is not None)
        graph.seq_id = seq_id
        return graph
