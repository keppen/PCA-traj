import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.analysis import align

from pathlib import Path
import pandas as pd
from MDAnalysis.analysis.dihedrals import Dihedral

# ---------------------------------------------------------------------
# CARTESIANS
# ---------------------------------------------------------------------


def _load_or_compute_cartesian(
    traj,
    output_prefix=None,
    npy_file=None,
    selection="all",
    align_to_first=True,
):
    """
    Prepare Cartesian coordinate matrix for PCA.
    Rows = frames
    Columns = 3 * N_atoms
    """

    # ---------------- load cached ----------------
    if npy_file:
        data_space = np.load(npy_file)
        if data_space.size == 0:
            raise ValueError("No data loaded!")
        return data_space

    universe = traj
    atoms = universe.select_atoms(selection)

    n_atoms = atoms.n_atoms
    n_frames = len(universe.trajectory)

    if n_atoms == 0:
        raise ValueError("Atom selection is empty!")

    print(f"Preparing Cartesian PCA data:")
    print(f"  atoms   : {n_atoms}")
    print(f"  frames  : {n_frames}")

    # ---------------- reference for alignment ----------------
    universe.trajectory[0]
    ref_coords = atoms.positions.copy()

    # storage
    coord_frames = np.zeros((n_frames, 3 * n_atoms))

    # ---------------- loop over trajectory ----------------
    for i, ts in enumerate(universe.trajectory):
        print(f" {i + 1}/{n_frames}", end="\r")

        if align_to_first:
            align.rotation_matrix(
                atoms.positions,
                ref_coords,
            )
            atoms.positions -= atoms.center_of_mass()

        coords = atoms.positions.copy()
        coord_frames[i, :] = coords.reshape(-1)

    print("\nCartesian coordinate extraction complete.")

    if output_prefix:
        out = f"cart-{output_prefix}.npy"
        np.save(out, coord_frames)
        print(f"Saved PCA input matrix to {out}")

    return coord_frames


def _get_sorted_labels_coords(coords, atoms):
    """
    Sort PCA coordinate matrix by resid and atom name with custom priority.

    Parameters
    ----------
    coords : np.ndarray
        Shape (3*n_atoms, n_frames)
    atoms : MDAnalysis.core.groups.AtomGroup
        AtomGroup corresponding to coords order

    Returns
    -------
    sorted_coords : np.ndarray
        Reordered coordinate matrix
    labels : list[str]
        Labels in format "resid:atomname" (one per atom)
    """

    n_atoms = len(atoms)
    assert coords.shape[0] == 3 * n_atoms

    # Custom atom priority
    priority_order = ["N", "HN", "CG", "CB", "OA", "C", "O"]
    priority_map = {name: i for i, name in enumerate(priority_order)}
    default_priority = len(priority_order)

    # Build sorting keys
    sort_keys = []
    for i, atom in enumerate(atoms):
        atom_priority = priority_map.get(atom.name, default_priority)
        sort_keys.append(
            (
                atom.resid,  # primary: mer id
                atom_priority,  # secondary: custom atom order
                atom.name,  # tertiary: alphabetical
                i,  # original atom index
            )
        )

    # Sort atoms
    sort_keys.sort()

    # ----- build row permutation -----
    row_perm = []
    labels = []

    for _, _, _, old_i in sort_keys:
        row_perm.extend([3 * old_i, 3 * old_i + 1, 3 * old_i + 2])

        labels.extend(
            [
                f"X:{atoms[old_i].resid}:{atoms[old_i].name}",
                f"Y:{atoms[old_i].resid}:{atoms[old_i].name}",
                f"Z:{atoms[old_i].resid}:{atoms[old_i].name}",
            ]
        )

    # ----- apply permutation on rows -----
    sorted_coords = coords[row_perm, :]

    return sorted_coords, labels


# ---------------------------------------------------------------------
# DISTANCES
# ---------------------------------------------------------------------


def _load_or_compute_distances(
    traj_sel,
    output_prefix,
    npy_file,
):
    if npy_file:
        data_space = np.load(npy_file)
        if data_space.size == 0:
            raise ValueError("No data loaded!")
        return data_space

    distance_frames = []

    universe = traj_sel.universe
    n_atoms = len(traj_sel)

    # Precompute index pairs (n, n+j) to avoid duplicates and self-pairs
    pair_indices = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]

    n_frames = len(universe.trajectory)

    for i, ts in enumerate(universe.trajectory):
        print(f" {i + 1}/{n_frames}", end="\r")

        coords = traj_sel.positions

        # Compute full distance matrix for this frame
        dist_matrix = distance_array(coords, coords)

        # Extract only (n, n+j) distances
        frame_distances = np.array([dist_matrix[i, j] for i, j in pair_indices])

        distance_frames.append(frame_distances)

    print("\nC–C distance calculation complete.")

    data_space = np.array(distance_frames)

    print(data_space)

    if output_prefix:
        np.save(f"dist-{output_prefix}.npy", data_space)

    return data_space


def _get_sorted_labels_dist(dist_data, atoms):
    """
    Sort distance data by atom pair with custom priority.

    Parameters
    ----------
    dist_data : np.ndarray
        Shape (n_frames, n_pairs)
    atoms : MDAnalysis.core.groups.AtomGroup

    Returns
    -------
    sorted_data : np.ndarray
        Reordered distance matrix (columns permuted)
    labels : list[str]
        Labels in format "resid1:atom1-resid2:atom2"
    """

    n_atoms = len(atoms)

    # Recreate pair indices in SAME order as generation
    pair_indices = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]

    assert dist_data.shape[0] == len(pair_indices), (
        "Mismatch between distance data and atom pairs"
    )

    # --- build sorting keys ---
    sort_keys = []

    for pair_idx, (i, j) in enumerate(pair_indices):
        ai = atoms[i]
        aj = atoms[j]

        key = (
            ai.resid,
            aj.resid,
            ai.name,
            aj.name,
            pair_idx,  # preserve stability
        )

        sort_keys.append(key)

    # --- sort ---
    sort_keys.sort()

    # --- build permutation ---
    col_perm = []
    labels = []

    for *_, pair_idx in sort_keys:
        i, j = pair_indices[pair_idx]

        col_perm.append(pair_idx)

        labels.append(f"{atoms[i].resid}-{atoms[j].resid}")

    # --- apply permutation (columns!) ---
    sorted_data = dist_data[col_perm, :]

    return sorted_data, labels


# ---------------------------------------------------------------------
# DIHEDRALS
# ---------------------------------------------------------------------


def compute_torsion_dataframe(ref_universe, nres, atom_selector_fn, verbose=True):
    """
    Select atoms for each residue using `atom_selector_fn` and compute dihedral angles.
    Returns a DataFrame of shape (n_frames, n_residues).
    """
    dihedral_groups = []
    for i in range(nres):
        sel = atom_selector_fn(ref_universe, i, nres)
        if not all(len(group) == 1 for group in sel):
            print(f"Residue {i} skipped due to missing atoms: {[len(g) for g in sel]}")
            continue
        dihedral_groups.append(sel[0] + sel[1] + sel[2] + sel[3])

    if not dihedral_groups:
        print("No valid dihedral groups found.")
        return pd.DataFrame()

    dih = Dihedral(dihedral_groups)
    dih.run(verbose=verbose, stop=-1)

    angles = dih.results.angles  # shape (n_frames, n_dihedrals)

    # Return DataFrame with 1 column per dihedral group
    return pd.DataFrame(angles)


# Custom torsion definitions for your polymer
def selector_torsion1(uni, i, nres):
    """PHI angle"""
    if i == 0:
        return [
            uni.select_atoms("resid 0 and name CT"),
            uni.select_atoms("resid 0 and name N"),
            uni.select_atoms("resid 0 and name CG"),
            uni.select_atoms("resid 0 and name CB"),
        ]
    else:
        return [
            uni.select_atoms(f"resid {i - 1} and name C"),
            uni.select_atoms(f"resid {i} and name N"),
            uni.select_atoms(f"resid {i} and name CG"),
            uni.select_atoms(f"resid {i} and name CB"),
        ]


def selector_torsion2(uni, i, nres):
    """XI angle"""
    return [
        uni.select_atoms(f"resid {i} and name N"),
        uni.select_atoms(f"resid {i} and name CG"),
        uni.select_atoms(f"resid {i} and name CB"),
        uni.select_atoms(f"resid {i} and name OA"),
    ]


def selector_torsion3(uni, i, nres):
    """CHI angle"""
    if i == nres - 1:
        next_atom = uni.select_atoms(f"resid {i} and name HO")
    else:
        next_atom = uni.select_atoms(f"resid {i} and name C")
    return [
        uni.select_atoms(f"resid {i} and name CG"),
        uni.select_atoms(f"resid {i} and name CB"),
        uni.select_atoms(f"resid {i} and name OA"),
        next_atom,
    ]


def _load_or_compute_dichedrals(
    traj,
    output_prefix,
    npy_file,
):
    if npy_file:
        data_space = np.load(npy_file)
        if data_space.size == 0:
            raise ValueError("No data loaded!")
        return data_space

    universe = traj
    n_atoms = len(traj.atoms)

    n_frames = len(universe.trajectory)

    nres = max(u.resid for u in universe.residues) + 1

    # Compute torsions
    torsion1 = compute_torsion_dataframe(
        universe, nres, selector_torsion1, verbose=True
    )
    sin_torsion1 = np.sin(np.deg2rad(torsion1))
    cos_torsion1 = np.cos(np.deg2rad(torsion1))
    print(torsion1)

    torsion2 = compute_torsion_dataframe(
        universe, nres, selector_torsion2, verbose=True
    )
    sin_torsion2 = np.sin(np.deg2rad(torsion2))
    cos_torsion2 = np.cos(np.deg2rad(torsion2))
    print(torsion2)

    torsion3 = compute_torsion_dataframe(
        universe, nres, selector_torsion3, verbose=True
    )
    sin_torsion3 = np.sin(np.deg2rad(torsion3))
    cos_torsion3 = np.cos(np.deg2rad(torsion3))
    print(torsion3)

    data = np.concatenate(
        [
            sin_torsion1,
            cos_torsion1,
            sin_torsion2,
            cos_torsion2,
            sin_torsion3,
            cos_torsion3,
        ],
        axis=1,
    )

    data_space = data
    print(data_space)

    if output_prefix:
        np.save(f"dihedral-{output_prefix}.npy", data_space)

    return data_space


def _get_sorted_labels_dihedrals(
    data,
):
    """
    Sort dihedral feature matrix by residue, torsion type, and trig component.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_frames, n_features)
    torsion*_df : pd.DataFrame
        Original torsion DataFrames BEFORE sin/cos expansion

    Returns
    -------
    sorted_data : np.ndarray
    labels : list[str]
    """

    # --- torsion metadata ---
    n_mers = int(data.shape[0] / 3)
    torsions = [
        ("PHI", data[:n_mers]),
        ("XI", data[n_mers : n_mers * 2]),
        ("CHI", data[n_mers * 2 :]),
    ]

    # Track column offsets in `data`
    col_offset = 0
    entries = []

    for torsion_name, array in torsions:
        n_cols = array.shape[0]

        # We assume columns correspond to residues in order,
        # but some residues may be skipped → use df.columns
        for local_idx, _ in enumerate(array):
            # sin column
            if local_idx < n_mers / 2:
                entries.append(
                    (
                        local_idx,
                        torsion_name,
                        "sin",
                        col_offset + local_idx,
                    )
                )

            # cos column
            if local_idx >= n_mers / 2:
                entries.append(
                    (
                        int(local_idx - (n_mers / 2)),
                        torsion_name,
                        "cos",
                        col_offset + local_idx,
                    )
                )

        col_offset += n_cols  # sin + cos block
    # print(entries)

    assert col_offset == data.shape[0], "Column mismatch in dihedral assembly"

    # --- sorting priority ---
    torsion_priority = {"PHI": 0, "XI": 1, "CHI": 2}
    trig_priority = {"sin": 0, "cos": 1}

    # --- sort ---
    entries.sort(
        key=lambda x: (
            x[0],  # resid
            torsion_priority[x[1]],
            trig_priority[x[2]],
        )
    )

    # --- build permutation ---
    col_perm = []
    labels = []

    for resid, torsion_name, trig, col_idx in entries:
        col_perm.append(col_idx)
        labels.append(f"{resid}:{torsion_name}:{trig}")
    print(labels)
    sorted_data = data[col_perm, :]

    # for i in range(sorted_data.shape[0]):
    #     print(i, data[i, 0], sorted_data[i, 0])

    return sorted_data, labels
