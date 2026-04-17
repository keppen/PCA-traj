import glob
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "boc-pgs4-chcl3"
# BASE_DIR = "10-mer-Kasper"
ENSEMBLES = ["cart", "dist", "dih"]
SELECTION = "not name H*"  # adjust if needed


# -----------------------------
# UTILITIES
# -----------------------------
def load_ensemble(path_pattern):
    files = sorted(glob.glob(path_pattern))
    universes = [mda.Universe(f) for f in files]
    return files, universes


def compute_rmsd(universes):
    rmsd_list = []

    # Use the first structure as reference

    for i, u in enumerate(universes):
        rmsd_tmp = []
        ref = u.select_atoms(SELECTION).positions
        for _ in u.trajectory:
            R = rms.rmsd(u.select_atoms(SELECTION).positions, ref, superposition=True)
            rmsd_tmp.append(R)
        rmsd_list.append(np.array(rmsd_tmp))
        # print(f"{files[i]} : RMSD = {R:.3f} Å")

    return rmsd_list


def find_medoid(universes):
    """
    Compute pairwise RMSD and return the index of the medoid.
    Medoid = structure with minimal sum of RMSDs to all others.
    """
    n_uni = len(universes)

    medoids = []

    for i in range(n_uni):
        n_geom = len(universes[i].trajectory)
        rmsd_mat = np.zeros((n_geom, n_geom))

        # store all coordinates first
        coords = [ts.positions.copy() for ts in universes[i].trajectory]

        for m, pos_m in enumerate(coords):
            for n, pos_n in enumerate(coords):
                print(f"m={m} n={n}", end="\r")
                R = rms.rmsd(pos_m, pos_n, superposition=True)
                rmsd_mat[m, n] = R
                rmsd_mat[n, m] = R  # symmetric

        sums = rmsd_mat.sum(axis=1)
        idx = np.argmin(sums)
        medoids.append(idx)
    return medoids


# -----------------------------
# PART 1: INTRA-ENSEMBLE RMSD
# -----------------------------
ensemble_data = {}

for ens in ENSEMBLES:
    pattern = f"{BASE_DIR}/{ens}/*.pdb"
    files, universes = load_ensemble(pattern)

    print(f"\n=== {ens.upper()} ===")
    print(f"Files: {len(files)}")

    # rmsd_data = compute_rmsd(universes)

    # print("Mean RMSD [Å]: ", "\t".join([f"{i.mean():.3f}" for i in rmsd_data]))
    # print("Std RMSD [Å]: ", "\t".join([f"{i.std():.3f}" for i in rmsd_data]))

    rmsd_data = 0

    # Store
    ensemble_data[ens] = {
        "files": files,
        "universes": universes,
        "rmsd_matrix": rmsd_data,
    }

# -----------------------------
# PART 2: MEDOID STRUCTURES
# -----------------------------
medoids = {}

for ens in ENSEMBLES:
    universes = ensemble_data[ens]["universes"]
    idx = find_medoid(universes)

    medoids[ens] = {
        "index": idx,
        "file": ensemble_data[ens]["files"][idx],
        "universe": universes[idx],
    }

    print(f"\nMedoid for {ens}:")
    print(f"Index: {idx}")
    print(f"File : {medoids[ens]['file']}")


# -----------------------------
# PART 3: FULL CROSS-ENSEMBLE RMSD MATRICES
# -----------------------------
print("\n=== MEDOID RMSD MATRIX ===")


def compute_cross_rmsd_matrix(universes1, universes2):
    n1 = len(universes1)
    n2 = len(universes2)

    mat = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            u1 = universes1[i]
            u2 = universes2[j]

            R = rms.rmsd(
                u1.select_atoms(SELECTION).positions,
                u2.select_atoms(SELECTION).positions,
                superposition=True,
            )

            mat[i, j] = R

    return mat


pairs = [
    ("cart", "dist"),
    ("cart", "dih"),
    ("dist", "dih"),
]

for e1, e2 in pairs:
    print(f"\n=== RMSD: {e1} vs {e2} ===")

    u1 = ensemble_data[e1]["universes"]
    u2 = ensemble_data[e2]["universes"]

    mat = compute_cross_rmsd_matrix(u1, u2)

    # pretty print
    header = "       " + " ".join(f"{e2}_{i + 1:>6}" for i in range(len(u2)))
    print(header)

    for i, row in enumerate(mat):
        row_str = " ".join(f"{v:6.2f}" for v in row)
        print(f"{e1}_{i + 1:>2}  {row_str}")

# # Pretty print
# print("\n    " + "  ".join(ENSEMBLES))
# for i, row in enumerate(mat):
#     print(f"{ENSEMBLES[i]}  " + "  ".join(f"{v:.2f}" for v in row))
