"""
src/data_utils.py
≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
Reusable EDA and visualization helpers for RNA 3D structure prediction.

All the fun utility functions that started life as inline code in the
notebook but have grown up and deserve a proper home here! 🎉

Covers
------
  - Toy structure generation  (make_helix)
  - Data quality checks       (check_data_quality)
  - Nucleotide statistics     (get_nt_composition)
  - Secondary structure       (greedy_wc_pairs, dot_bracket, count_pseudoknots)
  - Visualization             (draw_arc_plot, plot_contact_map,
                               rna_to_nx_graph, view_pdb_inline)
  - Per-residue RMSD analysis (per_residue_rmsd)
  - Minimal PDB writing       (coords_to_pdb_minimal)

Usage in notebook
-----------------
    from src.data_utils import (
        make_helix, greedy_wc_pairs, dot_bracket, count_pseudoknots,
        draw_arc_plot, plot_contact_map, rna_to_nx_graph,
        view_pdb_inline, per_residue_rmsd, check_data_quality,
        get_nt_composition, coords_to_pdb_minimal,
    )
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

# Optional heavy imports — handled gracefully so the module
# is still importable even on a minimal Kaggle environment.
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("seaborn not installed — contact-map heatmaps will use plain matplotlib")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    warnings.warn("networkx not installed — RNA graph visualisation unavailable")

try:
    import py3Dmol
    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False

# ── Colours — consistent palette used across ALL visualisations ───────────────
#  Blue=A, Tomato=U, Green=G, Purple=C
NT_COLORS: Dict[str, str] = {
    "A": "#2196F3",
    "U": "#FF5722",
    "G": "#4CAF50",
    "C": "#9C27B0",
}

# All Watson-Crick base pairs (including G-U wobble, common in RNA)
WC_PAIRS = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "U"), ("U", "G"),
}

# Default output directory for figures — created on first call
FIGURES_DIR = Path("./figures")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TOY STRUCTURE GENERATION                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def make_helix(
    L: int,
    radius: float = 3.0,
    pitch: float = 1.5,
    turns: Optional[float] = None,
) -> np.ndarray:
    """
    Generate idealised A-form RNA helical C3′ coordinates.

    Real A-form RNA has ~11 nucleotides per helical turn, a radius of ~9 Å
    and a rise of ~2.8 Å per residue. The defaults here are compressed for
    visual clarity in 2D/3D plots (not physically accurate!).

    Parameters
    ----------
    L      : Number of residues.
    radius : Helix radius in Å (default 3.0).
    pitch  : Rise per residue along z-axis in Å (default 1.5).
    turns  : Total number of helical turns. Defaults to L / 11.0.

    Returns
    -------
    coords : (L, 3) float64 array of (x, y, z) C3′ positions.
    """
    turns = turns if turns is not None else L / 11.0
    t = np.linspace(0, turns * 2 * np.pi, L)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), pitch * t])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DATA QUALITY CHECKS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def check_data_quality(
    df: pd.DataFrame,
    seq_col: str = "sequence",
    verbose: bool = True,
) -> dict:
    """
    Run three data-quality checks on a sequence DataFrame and return a summary.

    Checks performed
    ----------------
    1. **Missing values** — NaN cells AND empty strings ``""``
    2. **Invalid characters** — anything outside ``{A, U, G, C}``
    3. **FASTA artifacts** — ``>``, ``|``, embedded newlines in a cell

    Parameters
    ----------
    df      : DataFrame containing at least ``seq_col``.
    seq_col : Name of the nucleotide sequence column.
    verbose : Print results to stdout if True (default).

    Returns
    -------
    dict with keys::

        "n_nan"            : int — number of NaN cells
        "n_empty"          : int — number of empty-string cells
        "n_invalid_chars"  : int — number of sequences with non-AUGC characters
        "n_fasta_artifacts": int — number of sequences with FASTA scaffolding
        "all_clean"        : bool — True when all four counts are zero

    Examples
    --------
    >>> result = check_data_quality(df)
    >>> if not result["all_clean"]:
    ...     print("Fix your data before modelling!")
    """
    results: dict = {}

    # ── 1. Missing values ─────────────────────────────────────────────────────
    n_nan   = int(df[seq_col].isnull().sum())
    n_empty = int(df[seq_col].eq("").sum())
    results["n_nan"]   = n_nan
    results["n_empty"] = n_empty

    if verbose:
        print("=== Missing Value Check ===")
        if n_nan:
            print(f"  ⚠  NaN values in '{seq_col}': {n_nan} rows")
        else:
            print(f"  ✓  No NaN values in '{seq_col}'")
        if n_empty:
            print(f"  ⚠  Empty-string sequences: {n_empty} rows")
        else:
            print(f"  ✓  No empty-string sequences")

    # ── 2. Invalid characters ─────────────────────────────────────────────────
    valid_rna = re.compile(r"^[AUGCaugc]+$")
    invalid_mask = df[seq_col].notna() & ~df[seq_col].str.match(valid_rna)
    n_invalid = int(invalid_mask.sum())
    results["n_invalid_chars"] = n_invalid

    if verbose:
        print(f"\n=== Invalid Characters in '{seq_col}' ===")
        if n_invalid == 0:
            print("  ✓  All sequences contain only valid RNA characters {A, U, G, C}")
        else:
            print(f"  ⚠  {n_invalid} sequences contain unexpected characters:")
            for _, row in df[invalid_mask].head(5).iterrows():
                seq = str(row[seq_col])
                bad = sorted(set(re.sub(r"[AUGCaugc]", "", seq)))
                print(f"     bad_chars={bad}  snippet='{seq[:60]}'")
            if n_invalid > 5:
                print(f"     … and {n_invalid - 5} more")

    # ── 3. FASTA artifacts ────────────────────────────────────────────────────
    issues = {
        "contains >":       df[seq_col].str.contains(">",    na=False),
        "contains |":       df[seq_col].str.contains(r"\|",  regex=True, na=False),
        "contains newline": df[seq_col].str.contains(r"\n",  regex=True, na=False),
        "contains spaces":  df[seq_col].str.contains(r"\s",  regex=True, na=False),
    }
    n_artifacts = int(any(m.any() for m in issues.values()))
    results["n_fasta_artifacts"] = n_artifacts

    if verbose:
        print(f"\n=== Multi-chain / FASTA Artifact Detection ===")
        any_issue = False
        for flag, mask in issues.items():
            count = int(mask.sum())
            if count:
                any_issue = True
                print(f"  ⚠  '{flag}': {count} rows")
        if not any_issue:
            print("  ✓  No FASTA artifacts — sequences look like clean nucleotide strings")

    results["all_clean"] = (
        n_nan == 0 and n_empty == 0 and n_invalid == 0 and n_artifacts == 0
    )
    return results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NUCLEOTIDE STATISTICS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_nt_composition(
    seqs: pd.Series | List[str],
    normalise: bool = True,
) -> Dict[str, float]:
    """
    Compute A/U/G/C frequency across the entire sequence corpus.

    Parameters
    ----------
    seqs      : Series or list of RNA sequence strings.
    normalise : Return proportions (default) or raw counts.

    Returns
    -------
    dict mapping 'A'/'U'/'G'/'C' → frequency (0-1) or count.

    Examples
    --------
    >>> freq = get_nt_composition(df["sequence"])
    >>> print(freq)  # {'A': 0.258, 'U': 0.241, 'G': 0.289, 'C': 0.212}
    """
    all_chars = "".join(seqs)
    total = max(len(all_chars), 1)
    counts = {nt: all_chars.count(nt) for nt in "AUGC"}
    if normalise:
        return {nt: counts[nt] / total for nt in "AUGC"}
    return {nt: float(counts[nt]) for nt in "AUGC"}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECONDARY STRUCTURE HELPERS                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def greedy_wc_pairs(seq: str, min_loop: int = 3) -> List[Tuple[int, int]]:
    """
    Greedy O(N²) Watson-Crick base-pair finder — for visualisation only.

    Works by scanning left-to-right and pairing each residue with the
    nearest available complement (right-to-left search), respecting a
    minimum hairpin loop size of ``min_loop`` unpaired nucleotides.

    Handles A-U, G-C, and G-U wobble pairs (standard RNA).

    Parameters
    ----------
    seq      : RNA nucleotide string (case-insensitive).
    min_loop : Minimum number of residues between paired bases (default 3).
               Prevents pairs like ``i`` ↔ ``i+1`` which are impossible!

    Returns
    -------
    List of ``(i, j)`` index pairs with ``i < j``.

    Notes
    -----
    This is a *greedy* heuristic, **not** the thermodynamically optimal
    secondary structure (use RNAfold/Vienna for that in production).
    It is sufficient for visualisation tools like the arc plot.
    """
    seq = seq.upper()
    used: set = set()
    pairs: List[Tuple[int, int]] = []
    for i in range(len(seq)):
        for j in range(len(seq) - 1, i + min_loop, -1):
            if i in used or j in used:
                continue
            if (seq[i], seq[j]) in WC_PAIRS:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
    return pairs


def dot_bracket(seq: str, pairs: List[Tuple[int, int]]) -> str:
    """
    Convert a list of base pairs to dot-bracket notation.

    Dot-bracket is the industry-standard 1D encoding of RNA secondary
    structure:  ``(`` = 5′ end of a pair, ``)`` = 3′ end, ``.`` = unpaired.

    Parameters
    ----------
    seq   : RNA sequence string (used only for length).
    pairs : List of ``(i, j)`` pairs from :func:`greedy_wc_pairs`.

    Returns
    -------
    Dot-bracket string of the same length as ``seq``.

    Examples
    --------
    >>> dot_bracket("AUGCGAUC", [(0, 7), (1, 6)])
    '((..))..'
    """
    db = ["."] * len(seq)
    for i, j in pairs:
        db[i] = "("
        db[j] = ")"
    return "".join(db)


def count_pseudoknots(pairs: List[Tuple[int, int]]) -> int:
    """
    Count the number of pseudoknot-forming base pairs.

    A pseudoknot arises when arcs *cross*: pair ``(i, j)`` and ``(k, l)``
    are a pseudoknot if ``i < k < j < l``.  These are non-nested pairs that
    most secondary-structure prediction tools (RNAfold, Mfold, etc.) **cannot
    predict** — a major motivation for machine-learning approaches!

    Parameters
    ----------
    pairs : List of ``(i, j)`` base-pair tuples.

    Returns
    -------
    Number of crossing pair combinations (a rough pseudoknot count).
    """
    pk = 0
    sorted_p = sorted(pairs)
    for a, (i, j) in enumerate(sorted_p):
        for k, l in sorted_p[a + 1:]:
            if i < k < j < l:   # crossing arcs ↔ pseudoknot topology
                pk += 1
    return pk


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ARC PLOT                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def draw_arc_plot(
    ax: "matplotlib.axes.Axes",
    seq: str,
    pairs: List[Tuple[int, int]],
    title: str = "",
    arc_color: str = "#FF9800",
    nt_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Draw a secondary-structure arc plot on a matplotlib ``Axes``.

    Each base pair is drawn as a semicircular arc below the sequence axis.
    Nucleotide dots are coloured by type (A=blue, U=tomato, G=green, C=purple).

    Parameters
    ----------
    ax        : Matplotlib ``Axes`` to draw on.
    seq       : RNA nucleotide string.
    pairs     : ``(i, j)`` base-pair list from :func:`greedy_wc_pairs`.
    title     : Plot title.  Pair count and pseudoknot estimate are appended.
    arc_color : Colour for arc lines (default warm orange ``#FF9800``).
    nt_colors : Override the default nucleotide colour mapping.

    Notes
    -----
    Residue indices are printed every 5 positions above the axis.
    The dot-bracket string and pseudoknot count appear in the subplot title.
    """
    clrs = nt_colors or NT_COLORS
    L = len(seq)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-(L // 2) - 1, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw nucleotide dots along the x-axis
    for i, nt in enumerate(seq.upper()):
        c = clrs.get(nt, "#999999")
        ax.plot(i, 0, "o", color=c, markersize=5, zorder=3)

    # Draw base-pair arcs below the axis
    for i, j in pairs:
        mid = (i + j) / 2.0
        arc = mpatches.Arc(
            (mid, 0), width=j - i, height=j - i,
            angle=0, theta1=180, theta2=360,
            color=arc_color, lw=1.2, alpha=0.75,
        )
        ax.add_patch(arc)

    # Annotate: residue index every 5, nucleotide letter below axis
    for i, nt in enumerate(seq.upper()):
        if i % 5 == 0:
            ax.text(i, 0.6, str(i), ha="center", fontsize=7, color="#555555")
        ax.text(
            i, -0.6, nt, ha="center", fontsize=6,
            color=clrs.get(nt, "#999999"), fontweight="bold",
        )

    db_str = dot_bracket(seq, pairs)
    pk     = count_pseudoknots(pairs)
    ax.set_title(
        f"{title}  (L={L})\n"
        f"Pairs={len(pairs)}  Pseudoknots≈{pk}\n"
        f"DB: {db_str[:60]}{'…' if len(db_str) > 60 else ''}",
        fontsize=9, pad=4,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONTACT MAP                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def plot_contact_map(
    coords: np.ndarray,
    seq: str,
    threshold: float = 8.0,
    ax_dist: Optional["matplotlib.axes.Axes"] = None,
    ax_bin: Optional["matplotlib.axes.Axes"] = None,
    title: str = "Contact Map",
) -> None:
    """
    Plot a C3′–C3′ distance matrix and binary contact map side-by-side.

    The distance matrix shows actual inter-residue distances (Å).
    The binary map thresholds at ``threshold`` Å — the standard structural
    biology definition of "in contact."

    Long-range contacts (``|i - j| > 20``) are annotated with red dots on
    the binary contact panel — these are the tertiary interactions (helix
    packing, kissing loops) that deep learning models must learn from sequence
    context alone.

    Parameters
    ----------
    coords    : (L, 3) C3′ coordinate array.
    seq       : RNA sequence string (used for length and labels).
    threshold : Distance cutoff for binary contacts in Å (default 8.0).
    ax_dist   : Axes for the distance heatmap (left panel).  If None, skipped.
    ax_bin    : Axes for the binary contact map (right panel). If None, skipped.
    title     : Base title — length and stats are appended automatically.

    Notes
    -----
    Upper triangle is greyed out (masked) to eliminate visual redundancy —
    contact maps are symmetric by definition (``d(i,j) == d(j,i)``).
    """
    L        = len(seq)
    dist_mat = cdist(coords, coords)                          # (L, L) pairwise Å
    mask_upper = np.triu(np.ones_like(dist_mat, dtype=bool), k=1)

    # ── Distance heatmap (lower triangle only) ────────────────────────────────
    if ax_dist is not None:
        if HAS_SEABORN:
            import seaborn as sns  # local import to avoid hard dependency at top
            sns.heatmap(
                dist_mat, ax=ax_dist, mask=mask_upper,
                cmap="viridis_r", vmin=0, vmax=dist_mat.max(),
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "C3′–C3′ distance (Å)", "shrink": 0.7},
                linewidths=0,
            )
        else:
            d_lower = np.where(~mask_upper, dist_mat, np.nan)
            im = ax_dist.imshow(d_lower, cmap="viridis_r", origin="upper")
            plt.colorbar(im, ax=ax_dist, label="C3′–C3′ distance (Å)", fraction=0.03)
        ax_dist.set_title(f"{title}\nC3′ Distance Matrix (lower triangle)", fontsize=10)
        ax_dist.set_xlabel("Residue"); ax_dist.set_ylabel("Residue")

    # ── Binary contact map (< threshold Å, lower triangle) ───────────────────
    if ax_bin is not None:
        contact = (dist_mat < threshold).astype(float)
        np.fill_diagonal(contact, 0)                          # remove self-contacts
        contact_lower = np.where(~mask_upper, contact, np.nan)

        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(
                contact_lower, ax=ax_bin, cmap="Blues", vmin=0, vmax=1,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": f"Contact (< {threshold} Å)", "shrink": 0.7},
                linewidths=0,
            )
        else:
            im = ax_bin.imshow(contact_lower, cmap="Blues", origin="upper",
                               vmin=0, vmax=1)
            plt.colorbar(im, ax=ax_bin, label=f"Contact (< {threshold} Å)", fraction=0.03)

        # Overlay long-range contacts as red dots
        for i in range(L):
            for j in range(i):
                if contact[i, j] and abs(i - j) > 20:
                    ax_bin.plot(j + 0.5, i + 0.5, "r.", markersize=1.5, alpha=0.6)

        density = contact.sum() / 2 / max(L * (L - 1) / 2, 1)
        lr_cnt  = sum(
            1 for i in range(L) for j in range(i)
            if contact[i, j] and abs(i - j) > 20
        )
        ax_bin.set_title(
            f"Binary Contacts  (cutoff {threshold} Å)\n"
            f"Density={density:.3f}  Long-range(|i-j|>20)={lr_cnt} (red dots)",
            fontsize=10,
        )
        ax_bin.set_xlabel("Residue"); ax_bin.set_ylabel("Residue")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  NETWORKX RNA GRAPH                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def rna_to_nx_graph(seq: str):
    """
    Build a NetworkX graph representation of an RNA sequence.

    Graph topology
    --------------
    * **Nodes** — one per nucleotide, coloured by type (A/U/G/C).
    * **Backbone edges** (``edge_type='backbone'``, grey, solid) —
      consecutive residues ``i → i+1``.
    * **Base-pair edges** (``edge_type='base_pair'``, orange, dashed) —
      Watson-Crick pairs from :func:`greedy_wc_pairs`.

    Parameters
    ----------
    seq : RNA nucleotide string.

    Returns
    -------
    ``networkx.Graph`` with node/edge attributes set for drawing.

    Raises
    ------
    ImportError
        If ``networkx`` is not installed.
    """
    if not HAS_NX:
        raise ImportError("networkx is required: pip install networkx")

    _nx = nx  # type: ignore[possibly-undefined]
    G = _nx.Graph()  # type: ignore[union-attr]
    for i, nt in enumerate(seq):
        G.add_node(i, nucleotide=nt, color=NT_COLORS.get(nt.upper(), "#999999"))

    # Backbone edges (sequential connectivity)
    for i in range(len(seq) - 1):
        G.add_edge(i, i + 1, edge_type="backbone", color="#888888", width=1.0)

    # Base-pair edges (Watson-Crick)
    for i, j in greedy_wc_pairs(seq):
        G.add_edge(i, j, edge_type="base_pair", color="#FF9800", width=2.0)

    return G


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PER-RESIDUE RMSD                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def per_residue_rmsd(
    pred: np.ndarray,
    true: np.ndarray,
) -> np.ndarray:
    """
    Compute per-residue L2 distance after Kabsch alignment.

    Unlike the global RMSD scalar, this function returns one value per
    residue — revealing *where* the model fails (loops, junctions, etc.)
    rather than just *how much*.

    Parameters
    ----------
    pred : (L, 3) predicted C3′ coordinates.
    true : (L, 3) ground-truth C3′ coordinates.

    Returns
    -------
    rmsd_per_res : (L,) per-residue distance in Å after optimal alignment.

    Notes
    -----
    Uses the Kabsch algorithm from :mod:`src.utils` if available,
    otherwise falls back to a local implementation.
    """
    try:
        # Prefer the shared implementation from src.utils to avoid duplication
        src_root = str(Path(__file__).parent.parent.resolve())
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        from src.utils import kabsch_align as _kabsch
    except ImportError:
        # Inline fallback (identical algorithm)
        def _kabsch(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            mobile_c = mobile - mobile.mean(0)
            target_c = target - target.mean(0)
            H = mobile_c.T @ target_c
            U, _, Vt = np.linalg.svd(H)
            if np.linalg.det(Vt.T @ U.T) < 0:
                Vt[-1] *= -1
            R = Vt.T @ U.T
            return mobile_c @ R.T + target.mean(0)

    pred_aligned = _kabsch(pred, true)
    return np.sqrt(np.sum((pred_aligned - true) ** 2, axis=1))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MINIMAL PDB WRITER                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def coords_to_pdb_minimal(coords: np.ndarray, seq: str) -> str:
    """
    Convert (L, 3) coordinates to a minimal PDB string (C3′ atoms only).

    This is a lightweight fallback for when ``src.utils.coords_to_pdb_string``
    is not available.  Uses the same ATOM record format that py3Dmol and
    BioPython PDBParser can read.

    Parameters
    ----------
    coords : (L, 3) C3′ coordinate array.
    seq    : RNA sequence string of length L.

    Returns
    -------
    PDB-format string.
    """
    nt_map = {"A": "  A", "U": "  U", "G": "  G", "C": "  C"}
    lines = []
    for i, (xyz, nt) in enumerate(zip(coords, seq)):
        resname = nt_map.get(nt.upper(), "  G")
        lines.append(
            f"ATOM  {i+1:5d}  C3'{resname} A{i+1:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INLINE 3D VIEWER                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def view_pdb_inline(
    pdb_string: str,
    width: int = 650,
    height: int = 420,
    spin: bool = True,
) -> None:
    """
    Render a PDB string inline using py3Dmol.

    Colour scheme:
    * Backbone cartoon — translucent grey
    * Base sticks — coloured by nucleotide (A=blue, U=tomato, G=green, C=purple)
    * C3′ spheres — orange (these are the atoms your model predicts!)

    Parameters
    ----------
    pdb_string : PDB-format string (from :func:`coords_to_pdb_minimal` or
                 ``src.utils.coords_to_pdb_string``).
    width      : Viewer width in pixels (default 650).
    height     : Viewer height in pixels (default 420).
    spin       : If True, auto-spin the molecule for 3D depth perception.

    Notes
    -----
    Silently prints an install hint if py3Dmol is not available.
    """
    if not HAS_PY3DMOL:
        print("py3Dmol is not installed.  Run:  pip install py3Dmol")
        return

    _py3dmol = py3Dmol  # type: ignore[possibly-undefined]
    view = _py3dmol.view(width=width, height=height)
    view.addModel(pdb_string, "pdb")

    # Semi-transparent cartoon backbone
    view.setStyle({"cartoon": {"color": "grey", "opacity": 0.35, "thickness": 0.4}})

    # Base sticks coloured per nucleotide
    for nt, hex_color in {
        "A": "0x2196F3",
        "U": "0xFF5722",
        "G": "0x4CAF50",
        "C": "0x9C27B0",
    }.items():
        view.addStyle(
            {"resn": nt},
            {"stick": {"colorscheme": "default", "radius": 0.12, "color": hex_color}},
        )

    # C3′ spheres — the prediction target atoms!
    view.addStyle({"atom": "C3'"}, {"sphere": {"color": "orange", "radius": 0.4}})

    view.zoomTo()
    if spin:
        view.spin(True)
    view.show()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE SAVING HELPER                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_figure(
    fig: "matplotlib.figure.Figure",
    name: str,
    figures_dir: Optional[Path] = None,
    dpi: int = 150,
) -> Path:
    """
    Save a matplotlib figure to ``./figures/`` and print a confirmation.

    Parameters
    ----------
    fig         : Matplotlib Figure to save.
    name        : Filename (e.g. ``"2c_contact_map.png"``).
    figures_dir : Override output directory (default ``./figures``).
    dpi         : Output resolution (default 150 dpi — publication-quality).

    Returns
    -------
    Absolute path to the saved file.
    """
    out_dir = figures_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {out_path}")
    return out_path
