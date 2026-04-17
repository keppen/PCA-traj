import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np

# ---------------------------------------------------------------------
# PLOTTING: FREE ENERGY SURFACE
# ---------------------------------------------------------------------


def _plot_free_energy_surface(reduced_data, n_frames, output_prefix):
    print("Plotting PES...")
    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    H[H == 0] = np.nan

    T = 300
    kb = 1.380649e-23
    Na = 6.02214076e23
    R = 8.3145

    DG = -Na * kb * T * np.log(H / n_frames) / 1000

    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.imshow(
        DG.T,
        origin="lower",
        aspect="auto",
        cmap="nipy_spectral",
    )
    fig.colorbar(cax, label="kJ/mol")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}-FES.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# EXPLAINED VARIANCE
# ---------------------------------------------------------------------


def _plot_explained_variance(explained_variance_ratio, output_prefix):
    print("Plotting explained variance ratio...")
    plt.figure()
    plt.bar(
        range(1, len(explained_variance_ratio) + 1),
        explained_variance_ratio,
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}-explained.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# LOADINGS
# ---------------------------------------------------------------------


def _plot_loadings(
    loadings,
    n_features,
    output_prefix,
    labels=None,
    max_labels=100,
    top_k=None,
    use_line=False,
):
    """
    Scalable plotting of PCA loadings.

    Parameters
    ----------
    loadings : np.ndarray
        Shape (n_components, n_features)
    n_features : int
    output_prefix : str
    labels : list[str], optional
    max_labels : int
        Maximum number of x tick labels to display
    top_k : int or None
        If set, highlight only top_k absolute loadings
    use_line : bool
        Use line plot instead of bar plot for dense data
    """

    print("Plotting loadings...")

    if labels:
        if len(labels) != n_features:
            raise ValueError("Number of features and labels do not match.")
        feature_names = np.array(labels)
    else:
        feature_names = np.array([f"F{i}" for i in range(n_features)])

    x = np.arange(n_features)

    for pc in (0, 1):
        print(f"PC {pc + 1}")

        values = loadings[pc]

        # --- dynamic width ---
        fig_width = max(12, min(0.15 * n_features, 60))
        plt.figure(figsize=(fig_width, 5))

        # --- plotting mode ---
        if use_line or n_features > 200:
            plt.plot(x, values)
        else:
            plt.bar(x, values)

        # --- highlight top-k ---
        if top_k:
            idx = np.argsort(np.abs(values))[-top_k:]
            plt.scatter(idx, values[idx], zorder=3)

            for i in idx:
                plt.text(i, values[i], feature_names[i], rotation=90, fontsize=8)

        # --- adaptive ticks ---
        if n_features <= max_labels:
            tick_idx = x
        else:
            step = int(np.ceil(n_features / max_labels))
            tick_idx = x[::step]

        plt.xticks(tick_idx, feature_names[tick_idx], rotation=90)

        plt.ylabel(f"PC{pc + 1} loading")
        plt.xlabel("Feature index")

        plt.tight_layout()
        plt.savefig(f"{output_prefix}-loadings-PC{pc + 1}.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------
# MINIMA DETECTION
# ---------------------------------------------------------------------


def _find_free_energy_minima(
    reduced_data, n_frames, threshold=10, radius_multiplier=2, n_minima=3
):
    print("Finding energy minima...")

    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    H[H == 0] = np.nan

    T = 300
    kb = 1.380649e-23
    Na = 6.02214076e23
    R = 8.3145
    beta = 1 / (R * T / 1000)

    DG = -Na * kb * T * np.log(H / n_frames) / 1000

    valid_mask = H > 0
    DG_masked = np.ma.masked_where(~valid_mask, DG)
    DG_smooth = gaussian_filter(DG_masked, sigma=0.1)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    DG_kJmol = np.ma.masked_where(~valid_mask, DG_smooth.copy())

    minima_pca_coords = []
    regions = []
    global_mask = np.zeros_like(DG_kJmol.mask, dtype=bool)

    def circular_mask(x_center, y_center, radius):
        return (X - x_center) ** 2 + (Y - y_center) ** 2 <= radius**2

    while True:
        DG_tmp = np.ma.masked_where(global_mask | DG_kJmol.mask, DG_kJmol)
        if DG_tmp.count() == 0:
            break

        min_idx = np.unravel_index(np.argmin(DG_tmp), DG_tmp.shape)
        x_min = x_centers[min_idx[0]]
        y_min = y_centers[min_idx[1]]

        minima_pca_coords.append((x_min, y_min))

        for radius in np.linspace(0.03, 3, 1000):
            mask = circular_mask(x_min, y_min, radius)
            masked_vals = DG_kJmol[mask]

            # print(
            #     f"radius {radius:.2f}, delta E {masked_vals.max() - masked_vals.min():.2f}, n masked {masked_vals.count()}"
            # )
            if masked_vals.count() < 5:
                continue
            if masked_vals.max() - masked_vals.min() > threshold:
                break

        region_mask = circular_mask(x_min, y_min, radius)
        regions.append(region_mask)

        global_mask |= circular_mask(x_min, y_min, radius * radius_multiplier)

        print(
            f"Min {len(minima_pca_coords)} Free energy = "
            f"{masked_vals.min():.2f} kJ/mol at PCA coords "
            f"({x_min:.3f}, {y_min:.3f})"
        )

        if len(minima_pca_coords) == n_minima:
            break

    return minima_pca_coords, regions


# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------


def _plot_fes_with_minima(
    reduced_data,
    minima_pca_coords,
    regions,
    output_prefix,
):
    print("Ploting PES with minima...")

    H, xedges, yedges = np.histogram2d(
        reduced_data[:, 0],
        reduced_data[:, 1],
        bins=250,
    )

    H[H == 0] = np.nan

    T = 300
    kb = 1.380649e-23
    Na = 6.02214076e23

    DG = -Na * kb * T * np.log(H / len(reduced_data)) / 1000
    DG_smooth = gaussian_filter(DG, sigma=0.1)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    plt.figure(figsize=(17 / 2.5, 4))

    plt.contourf(
        x_centers,
        y_centers,
        DG_smooth.T,
        levels=100,
        cmap="nipy_spectral",
    )
    cbar = plt.colorbar(label="Free energy [kJ/mol]")
    cbar.set_label("Free energy [kJ/mol]", rotation=270, labelpad=15)

    for i, (x, y) in enumerate(minima_pca_coords):
        plt.plot(x, y, "wo")
        plt.text(x + 0.03, y + 0.03, f"Min {i + 1}", fontsize=9, weight="bold")

    for (x, y), mask in zip(minima_pca_coords, regions):
        plt.contour(
            x_centers,
            y_centers,
            mask.T.astype(int),
            levels=[0.5],
            colors="white",
            linewidths=1.5,
        )
        plt.plot(x, y, "wo")

    plt.xlabel("Principal Component I")
    plt.ylabel("Principal Component II")
    plt.title("Free Energy Surface with Minima")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}-FES-smoothed.png", dpi=1000)
    plt.close()


def plot_feature_distributions(
    data_space,
    feature_indices=None,
    n_bins=50,
    max_features=20,
    figsize=(10, 6),
    title="Feature Distributions",
):
    """
    Plot distributions (histograms) of selected features.

    Parameters
    ----------
    data_space : np.ndarray
        Shape (n_features, n_samples)

    feature_indices : list[int] or None
        Indices of features to plot. If None, the first `max_features`
        features are plotted.

    n_bins : int
        Number of histogram bins.

    max_features : int
        Maximum number of features to plot if feature_indices is None.

    figsize : tuple
        Matplotlib figure size.

    title : str
        Figure title.
    """
    print("Ploting feature distributions...")

    n_features, n_samples = data_space.shape

    if feature_indices is None:
        feature_indices = list(range(min(n_features, max_features)))

    plt.figure(figsize=figsize)

    for idx in feature_indices:
        values = data_space[idx]

        plt.hist(
            values,
            bins=n_bins,
            density=True,
            alpha=0.5,
            label=f"Feature {idx}",
        )

    plt.xlabel("Value")
    plt.ylabel("Probability density")
    plt.title(title)

    if len(feature_indices) <= 10:
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"data_structure.png", dpi=200)
    plt.close()


def plot_data_structure(
    data_space,
    figsize=(10, 6),
    title="Data structure",
    max_features=5,
    feature_indices=None,
):
    """
    Plot distributions (histograms) of selected features.

    Parameters
    ----------
    data_space : np.ndarray
        Shape (n_features, n_samples)

    feature_indices : list[int] or None
        Indices of features to plot. If None, the first `max_features`
        features are plotted.

    n_bins : int
        Number of histogram bins.

    max_features : int
        Maximum number of features to plot if feature_indices is None.

    figsize : tuple
        Matplotlib figure size.

    title : str
        Figure title.
    """
    print(f"Plotting data structure - {title}")
    n_features, n_samples = data_space.shape

    if feature_indices is None:
        feature_indices = list(range(min(n_features, max_features)))

    plt.figure(figsize=figsize)

    for idx in feature_indices:
        values = data_space[idx]
        lim = 100000

        plt.plot(
            range(n_samples)[:lim],
            values[:lim],
            alpha=0.5,
            label=f"Feature {idx}",
        )

    plt.xlabel("Value")
    plt.ylabel("Sample")
    plt.title(title)

    if len(feature_indices) <= 10:
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{title}-data-structure.png", dpi=200)
    plt.close()
