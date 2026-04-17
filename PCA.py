from time import time
import sys
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align

from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plotting import (
    _plot_free_energy_surface,
    _plot_explained_variance,
    _plot_loadings,
    _find_free_energy_minima,
    _plot_fes_with_minima,
    plot_feature_distributions,
    plot_data_structure,
)
from DataExtraction import (
    _load_or_compute_distances,
    _load_or_compute_dichedrals,
    _load_or_compute_cartesian,
    _get_sorted_labels_coords,
    _get_sorted_labels_dihedrals,
    _get_sorted_labels_dist,
)

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
DATA_TYPE = None
NPY_FILE = None
NPY_FILE = None
NPY_FILE = None

INPUT_DIR = Path(sys.argv[1])
NAMED_PDB = Path(sys.argv[2])  # topology
DATA_TYPE = sys.argv[3]


if "--npy" in sys.argv:
    NPY_FILE = sys.argv[sys.argv.index("--npy") + 1]

if "--npy" in sys.argv:
    NPY_FILE = sys.argv[sys.argv.index("--npy") + 1]

if "--npy" in sys.argv:
    NPY_FILE = sys.argv[sys.argv.index("--npy") + 1]

GLOB = "*.xtc"
THRESHOLD = 8  # for finding minima, depth of minimum in kj/mol
RADIUS_MULTIPLIER = 2  # for finding minima, change
N_MINIMA = 5  # for finding minima, number of minima
SELECTION = "name OA"
OUTPUT_PREFIX = "pca"


# ---------------------------------------------------------------------
# CORE WORKFLOW
# ---------------------------------------------------------------------


def universe_setup(
    traj_files,
    top_file: Path,
    output_prefix: str = "pca",
    dist_selection: str = "name N HN O OA C CB* CG*",
):
    """
    Compute RMSD of trajectory to each middle structure,
    then perform PCA on the RMSD feature space.
    """

    start_time = time()

    traj_uni = _load_universes(top_file, traj_files)
    if DATA_TYPE == "DIST":
        traj_sel = traj_uni.select_atoms(dist_selection)

        n_frames = len(traj_uni.trajectory)

        print(f"Trajectory frames : {n_frames}")

        dist_data_space = _load_or_compute_distances(
            traj_sel, output_prefix, NPY_FILE
        ).T
        print(dist_data_space.shape)
        sorted_dist_data_space, sorted_labels = _get_sorted_labels_dist(
            dist_data_space, traj_sel
        )

        data_space = sorted_dist_data_space
        labels = sorted_labels

    elif DATA_TYPE == "DIH":
        n_frames = len(traj_uni.trajectory)
        print(f"Trajectory frames : {n_frames}")

        dih_data_space = _load_or_compute_dichedrals(
            traj_uni, output_prefix, NPY_FILE
        ).T
        print(dih_data_space.shape)
        sorted_dih_data_space, sorted_labels = _get_sorted_labels_dihedrals(
            dih_data_space
        )

        # for i in range(len(sorted_labels)):
        #     print(f"{sorted_labels[i]} - {sorted_dih_data_space[:, 0][i]}")

        data_space = sorted_dih_data_space
        labels = sorted_labels
        # labels = None

        print(data_space.shape)
    elif DATA_TYPE == "CART":
        selection = "all"
        atoms = traj_uni.select_atoms(selection)

        n_frames = len(traj_uni.trajectory)
        print(f"Trajectory frames : {n_frames}")

        cart_data_space = _load_or_compute_cartesian(
            traj_uni, output_prefix, NPY_FILE, selection=selection
        ).T

        sorted_cart_data_space, sorted_labels = _get_sorted_labels_coords(
            cart_data_space, atoms
        )

        for i in range(len(sorted_labels)):
            print(f"{sorted_labels[i]} - {sorted_cart_data_space[:, 0][i]}")

        data_space = sorted_cart_data_space
        labels = sorted_labels
        print(data_space.shape)
    else:
        print("No datatype set up. Aborting.")
        exit(1)

    plot_feature_distributions(
        data_space,
        max_features=15,
        title="Distance Feature Distributions",
    )

    plot_data_structure(
        data_space,
        title="pre-solvgol",
        max_features=5,
        feature_indices=None,
    )

    # _apply_time_filter(data_space)

    # plot_data_structure(
    #     data_space,
    #     title="post-solvgol",
    #     max_features=5,
    #     feature_indices=None,
    # )

    reduced_data, explained_variance_ratio, pca = _run_pca(data_space)

    _plot_free_energy_surface(
        reduced_data,
        n_frames,
        output_prefix,
    )

    _plot_explained_variance(
        explained_variance_ratio,
        output_prefix,
    )

    _plot_loadings(
        pca.components_,
        data_space.shape[0],
        output_prefix,
        labels=labels,
    )

    minima_pca_coords, regions = _find_free_energy_minima(
        reduced_data,
        n_frames,
        THRESHOLD,
        RADIUS_MULTIPLIER,
        N_MINIMA,
    )

    _plot_fes_with_minima(
        reduced_data,
        minima_pca_coords,
        regions,
        output_prefix,
    )

    _save_structure_indices(
        reduced_data,
        regions,
    )

    save_average_structures_per_minimum(
        traj_uni,
        "all",
        reduced_data,
        regions,
        output_prefix=OUTPUT_PREFIX,
    )

    print(f"[DONE] Total runtime: {time() - start_time:.2f} s")


# ---------------------------------------------------------------------
# UNIVERSE SETUP
# ---------------------------------------------------------------------


def _load_universes(top_file, traj_files):
    print("[START] Loading universes")
    traj_uni = mda.Universe(top_file, traj_files)
    return traj_uni


# ---------------------------------------------------------------------
# FILTERING
# ---------------------------------------------------------------------


def _apply_time_filter(data_space):
    print("Applying Savitzky–Golay filtering...", end="\r")

    for i in range(data_space.shape[0]):
        data_space[i] = savgol_filter(
            data_space[i],
            window_length=50,
            polyorder=9,
        )

    print("done.")


# ---------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------


def _run_pca(data_space):
    print("Running PCA...", end="\r")

    X = data_space.T
    scaler = StandardScaler(
        with_std=False
    )  # tlyko odejmowanie średniej. Dzielenie przez odchylenie std. odbiera fizycznosc analizy
    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        whiten=False
    )  # bez normalizowania wariancji. Pozbywa się wtedy fizycznej interpetacji.
    reduced_data = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_

    print("done.")
    print("Explained variance:", explained_variance_ratio[:5])

    return reduced_data, explained_variance_ratio, pca


# ---------------------------------------------------------------------
# STRUCTURE EXTRACTION
# ---------------------------------------------------------------------


def _save_structure_indices(reduced_data, regions):
    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    x_bin_idx = np.digitize(reduced_data[:, 0], xedges) - 1
    y_bin_idx = np.digitize(reduced_data[:, 1], yedges) - 1

    x_bin_idx = np.clip(x_bin_idx, 0, len(x_centers) - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, len(y_centers) - 1)

    for i, region_mask in enumerate(regions):
        idx = []
        for j in range(reduced_data.shape[0]):
            if region_mask[x_bin_idx[j], y_bin_idx[j]]:
                idx.append(j)

        idx = np.array(idx, dtype=int)
        print(f"Min {i + 1}: Saving {len(idx)} structures.")
        np.save(f"pca-idx-min{i + 1}.npy", idx)


def save_average_structures_per_minimum(
    universe,
    atom_selection,
    reduced_data,
    regions,
    output_prefix,
):
    """
    For each PCA minimum region:
      - collect all structures belonging to that region
      - compute the average structure
      - compute RMSD of each structure to the average
      - save the average structure and RMSD statistics
    """

    # Histogram definition (must match region construction)
    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    x_bin_idx = np.digitize(reduced_data[:, 0], xedges) - 1
    y_bin_idx = np.digitize(reduced_data[:, 1], yedges) - 1

    x_bin_idx = np.clip(x_bin_idx, 0, len(x_centers) - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, len(y_centers) - 1)

    sel = universe.select_atoms(atom_selection)
    n_atoms = sel.n_atoms

    for i, region_mask in enumerate(regions):
        frame_indices = []

        for j in range(reduced_data.shape[0]):
            if region_mask[x_bin_idx[j], y_bin_idx[j]]:
                frame_indices.append(j)

        frame_indices = np.array(frame_indices, dtype=int)
        print(frame_indices)

        if frame_indices.size == 0:
            print(f"Min {i + 1}: no structures found, skipping.")
            continue

        print(f"Min {i + 1}: Averaging {len(frame_indices)} structures.")

        # ------------------------------------------------------------
        # COLLECT COORDINATES
        # ------------------------------------------------------------

        # coords = np.zeros((len(frame_indices), n_atoms, 3), dtype=np.float64)

        # ------------------------------------------------------------
        # REFERENCE STRUCTURE (first frame in minimum)
        # ------------------------------------------------------------

        universe.trajectory[frame_indices[0]]
        ref_atoms = sel.copy()  # AtomGroup copy
        ref_coords = ref_atoms.positions.copy()

        # ------------------------------------------------------------
        # COLLECT ALIGNED COORDINATES
        # ------------------------------------------------------------

        # for k, frame in enumerate(frame_indices):
        #     universe.trajectory[frame]
        #
        #     # Align sel (mobile) to ref_atoms (reference)
        #     align.alignto(
        #         sel,
        #         ref_atoms,
        #         weights=None,
        #     )
        #
        #     coords[k] = sel.positions.copy()

        # ------------------------------------------------------------
        # AVERAGE STRUCTURE
        # ------------------------------------------------------------

        # avg_coords = coords.sum(axis=0) / len(frame_indices)
        # print(avg_coords)

        # ------------------------------------------------------------
        # RMSD TO AVERAGE
        # ------------------------------------------------------------

        # rmsd_vals = np.zeros(len(frame_indices), dtype=np.float64)
        #
        # ref_coords = avg_coords.copy()
        #
        # for k in range(len(frame_indices)):
        #     diff = coords[k] - ref_coords
        #     rmsd_vals[k] = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        #
        # rmsd_mean = rmsd_vals.mean()
        # rmsd_std = rmsd_vals.std()
        #
        # print(f"Min {i + 1}: RMSD to average = {rmsd_mean:.3f} ± {rmsd_std:.3f} Å")

        # ------------------------------------------------------------
        # SAVE AVERAGE STRUCTURE
        # ------------------------------------------------------------

        # universe.trajectory[frame_indices[0]]
        # sel.positions = avg_coords
        #
        # pdb_name = f"{output_prefix}-avg-min{i + 1}.pdb"
        #
        # sel.write(
        #     pdb_name,
        #     bonds="conect",
        # )

        # ------------------------------------------------------------
        # SAVE ALL FRAMES FROM THIS MINIMUM AS ONE MULTI-MODEL PDB
        # ------------------------------------------------------------

        pdb_name = f"{output_prefix}-ensemble-min{i + 1}.pdb"

        with mda.Writer(pdb_name, sel.n_atoms, bonds="conect") as W:
            for k, frame in enumerate(frame_indices):
                if k % 10 != 0:
                    continue

                print(f"Progress: {k} / {len(frame_indices)}", end="\r")
                universe.trajectory[frame]

                # Align to reference again (same as above)
                align.alignto(
                    sel,
                    ref_atoms,
                    weights=None,
                )

                W.write(sel)


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    files = sorted(INPUT_DIR.glob(GLOB))

    if not files:
        print(f"No files found in {INPUT_DIR} matching {GLOB}")
        sys.exit(1)

    print(f"Found {len(files)} trajectory files.")

    universe_setup(
        files,
        NAMED_PDB,
        dist_selection=SELECTION,
        output_prefix=OUTPUT_PREFIX,
    )
