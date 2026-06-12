"""
estimate.py — Signal-adaptive plot boundary alignment.

Architecture
============

  0. Signal census  — measure boundary.tif coverage per plot, decide
                       which signal regime the village is in:
                         STRONG  : hint_density > 0.03  (use bnd heavily)
                         WEAK    : 0.002 < density ≤ 0.03 (combine bnd + rgb)
                         ABSENT  : density ≤ 0.002  (rgb / Canny only)

  1. Global shift   — FFT phase correlation, but ONLY on STRONG-regime plots.
                       If too few strong plots exist (< MIN_STRONG_VOTES),
                       global shift is declared UNKNOWN and Stage 2 is skipped.

  2. Rubbersheeting — KNN local shifts from ~150 sample plots, filtered to
                       STRONG + WEAK regime only.
                       Falls back gracefully to zero-shift field when no
                       reliable samples are found.

  3. Fine alignment — per-plot, signal-regime-aware:
                         STRONG  → boundary edge recall (bnd_grid)
                         WEAK    → 0.6 * bnd_grid + 0.4 * rgb_grid
                         ABSENT  → rgb_grid only (Canny + Sobel gradient energy)

Why this matters
================
For Vadnerbhairav (dense bnd coverage): behaves identically to the previous
version — almost all plots are STRONG, global shift is well-determined.

For Malatavadi (sparse bnd coverage): most plots fall into WEAK or ABSENT.
The global shift either isn't attempted (too few strong votes) or is
computed only from the small number of plots that DO have good coverage.
Per-plot alignment then uses imagery directly — so zero-coverage plots
still get a meaningful correction rather than inheriting a poisoned shift.
"""

import numpy as np
import geopandas as gpd
from pathlib import Path
import time
from scipy.ndimage import sobel as scipy_sobel
from shapely.affinity import translate
from pyproj import Transformer
from shapely.ops import transform as shp_transform

from bhume import load, write_predictions
from bhume.geo import patch_for_plot, open_imagery, geom_to_imagery_crs, Patch
from rasterio.windows import from_bounds
from rasterio.features import rasterize


# ---------------------------------------------------------------------------
# Tuning constants — change these if your pixel size or village density differs
# ---------------------------------------------------------------------------

DENSITY_STRONG   = 0.03    # hint pixels / total pixels — use boundary heavily
DENSITY_WEAK     = 0.002   # hint pixels / total pixels — combine bnd + rgb
                            # below DENSITY_WEAK → ABSENT, rgb only

MIN_STRONG_VOTES = 10       # minimum STRONG-regime plots to trust global shift
GLOBAL_MAX_PIX   = 20       # discard phase-corr votes larger than this (outliers)
GLOBAL_N_SAMPLE  = 150       # how many plots to sample for global estimation
RUBBER_N_SAMPLE  = 150      # how many plots to sample for rubbersheeting
RUBBER_SEARCH    = 5        # pixel search radius for rubbersheeting
FINE_SEARCH      = 4        # pixel search radius for fine per-plot alignment


# ---------------------------------------------------------------------------
# Signal regime classifier
# ---------------------------------------------------------------------------

def hint_density(bnd_patch: "Patch") -> float:
    """Fraction of pixels in the boundary patch that are non-zero."""
    total = bnd_patch.image.size
    if total == 0:
        return 0.0
    return float((bnd_patch.image > 0).sum()) / total


def signal_regime(density: float) -> str:
    """Return 'strong', 'weak', or 'absent' based on boundary hint density."""
    if density > DENSITY_STRONG:
        return "strong"
    if density > DENSITY_WEAK:
        return "weak"
    return "absent"


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def patch_for_plot_single_band(src, geom_4326, pad_m: float = 25.0) -> Patch:
    """Read a single-band image crop covering a plot (e.g. for boundaries.tif)."""
    g = geom_to_imagery_crs(src, geom_4326)
    minx, miny, maxx, maxy = g.bounds
    left   = minx - pad_m
    bottom = miny - pad_m
    right  = maxx + pad_m
    top    = maxy + pad_m

    dl, db, dr, dt = src.bounds
    left   = max(left,   dl)
    bottom = max(bottom, db)
    right  = min(right,  dr)
    top    = min(top,    dt)
    if right <= left or top <= bottom:
        raise ValueError("plot bounding box does not overlap the imagery extent")

    window = from_bounds(left, bottom, right, top, transform=src.transform)
    gray   = src.read(1, window=window)
    image  = np.expand_dims(gray, axis=-1)
    return Patch(
        image=image,
        transform=src.window_transform(window),
        crs=str(src.crs),
        bounds=(left, bottom, right, top),
    )


def polygon_mask(poly, patch) -> np.ndarray:
    """Rasterize a polygon into patch pixel coordinates (filled uint8)."""
    return rasterize(
        [(poly, 1)],
        out_shape=(patch.image.shape[0], patch.image.shape[1]),
        transform=patch.transform,
        fill=0,
        dtype=np.uint8,
    )


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a filled polygon mask into a 1-pixel-wide boundary mask."""
    edge = np.zeros_like(mask)
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return edge.astype(np.uint8)


def shift_mask(mask: np.ndarray, dx_pix: int, dy_pix: int) -> np.ndarray:
    """Shift a 2-D binary mask by (dx_pix, dy_pix), zero-padding the vacated region."""
    H, W = mask.shape
    shifted = np.zeros_like(mask)
    y_dst_s = max(0, dy_pix)
    y_dst_e = min(H, H + dy_pix)
    y_src_s = max(0, -dy_pix)
    y_src_e = min(H, H - dy_pix)

    x_dst_s = max(0, dx_pix)
    x_dst_e = min(W, W + dx_pix)
    x_src_s = max(0, -dx_pix)
    x_src_e = min(W, W - dx_pix)

    if (y_src_e > y_src_s) and (x_src_e > x_src_s):
        shifted[y_dst_s:y_dst_e, x_dst_s:x_dst_e] = \
            mask[y_src_s:y_src_e, x_src_s:x_src_e]
    return shifted


def overlap_score(a: np.ndarray, b: np.ndarray, base_sum: float) -> float:
    """Fraction of 'a' pixels that overlap with 'b' (boundary recall)."""
    return float(np.logical_and(a, b).sum()) / base_sum if base_sum > 0 else 0.0


# ---------------------------------------------------------------------------
# Stage 0 — Village signal census
# ---------------------------------------------------------------------------

def village_signal_census(village, bnd_src, n_sample: int = 80) -> dict:
    """
    Sample up to n_sample plots and measure their boundary hint density.

    Returns a dict with:
      densities        : list of float
      regimes          : list of 'strong'/'weak'/'absent'
      n_strong / n_weak / n_absent : counts
      dominant_regime  : the most common regime
      mean_density     : mean of all densities
    """
    step = max(1, len(village.plots) // n_sample)
    sampled = village.plots.iloc[::step]

    densities = []
    regimes   = []

    for _, row in sampled.iterrows():
        try:
            patch = patch_for_plot_single_band(bnd_src, row.geometry, pad_m=15.0)
            d = hint_density(patch)
            densities.append(d)
            regimes.append(signal_regime(d))
        except Exception:
            densities.append(0.0)
            regimes.append("absent")

    n_strong = regimes.count("strong")
    n_weak   = regimes.count("weak")
    n_absent = regimes.count("absent")
    total    = len(regimes)

    dominant = max(["strong", "weak", "absent"], key=lambda r: regimes.count(r))

    census = {
        "densities":       densities,
        "regimes":         regimes,
        "n_strong":        n_strong,
        "n_weak":          n_weak,
        "n_absent":        n_absent,
        "n_total":         total,
        "dominant_regime": dominant,
        "mean_density":    float(np.mean(densities)) if densities else 0.0,
        "frac_strong":     n_strong / total if total else 0.0,
        "frac_absent":     n_absent / total if total else 0.0,
    }

    print(
        f"  Census ({total} plots sampled): "
        f"strong={n_strong} ({100*n_strong//total}%)  "
        f"weak={n_weak} ({100*n_weak//total}%)  "
        f"absent={n_absent} ({100*n_absent//total}%)  "
        f"mean_density={census['mean_density']:.4f}  "
        f"dominant={dominant}"
    )
    return census


# ---------------------------------------------------------------------------
# Stage 1 — Global shift via FFT phase correlation (strong plots only)
# ---------------------------------------------------------------------------

def _phase_correlation_shift(
    ref: np.ndarray, mov: np.ndarray
) -> tuple[float, float, float]:
    """
    Normalised cross-power spectrum phase correlation.
    Returns (dy_pix, dx_pix, peak_confidence).
    Confidence = peak / mean of correlation surface (higher = sharper).
    """
    R   = np.fft.fft2(ref)
    M   = np.fft.fft2(mov)
    eps = 1e-8
    xp  = (R * np.conj(M)) / (np.abs(R * np.conj(M)) + eps)
    corr = np.abs(np.fft.ifft2(xp))

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = corr[peak_idx]
    conf     = peak_val / (corr.mean() + eps)

    H, W = corr.shape
    dy = float(peak_idx[0] if peak_idx[0] < H // 2 else peak_idx[0] - H)
    dx = float(peak_idx[1] if peak_idx[1] < W // 2 else peak_idx[1] - W)
    return dy, dx, float(conf)


def estimate_global_shift(
    village, bnd_src, census: dict, pixel_size_x: float, pixel_size_y: float
) -> tuple[float, float, int, int, bool]:
    """
    Estimate the village-wide GPS offset using FFT phase correlation.

    Only uses plots classified as STRONG (hint_density > DENSITY_STRONG).
    If fewer than MIN_STRONG_VOTES strong plots are found, returns
    (0, 0, 0, 0, False) — the caller should skip rubbersheeting and
    rely on per-plot imagery alignment only.

    Returns (global_dx_m, global_dy_m, global_dx_pix, global_dy_pix, is_valid).
    """
    step    = max(1, len(village.plots) // GLOBAL_N_SAMPLE)
    sampled = village.plots.iloc[::step]

    dx_votes:   list[float] = []
    dy_votes:   list[float] = []
    conf_votes: list[float] = []

    for _, row in sampled.iterrows():
        try:
            patch = patch_for_plot_single_band(bnd_src, row.geometry, pad_m=40.0)

            # Only use STRONG-regime plots for global estimation
            d = hint_density(patch)
            if signal_regime(d) != "strong":
                continue

            H, W = patch.image.shape[0], patch.image.shape[1]
            if H < 32 or W < 32:
                continue

            plot_geom_u = geom_to_imagery_crs(bnd_src, row.geometry)
            ref_mask    = boundary_mask(polygon_mask(plot_geom_u, patch)).astype(np.float32)
            hint_f      = (patch.image[:, :, 0] > 0).astype(np.float32)

            if ref_mask.sum() == 0 or hint_f.sum() == 0:
                continue

            dy_pix, dx_pix, conf = _phase_correlation_shift(ref_mask, hint_f)

            if abs(dx_pix) > GLOBAL_MAX_PIX or abs(dy_pix) > GLOBAL_MAX_PIX:
                continue

            dx_votes.append(dx_pix * pixel_size_x)
            dy_votes.append(dy_pix * pixel_size_y)
            conf_votes.append(conf)

        except Exception:
            continue

    n_votes = len(dx_votes)
    print(f"  Global shift: {n_votes} strong-plot votes collected")

    if census["mean_density"] < 0.04:
        print(
            f"  WARNING: mean density {census['mean_density']:.4f} is too low (< 0.04). "
            f"Global shift declared UNKNOWN to avoid matching parallel noise."
        )
        return 0.0, 0.0, 0, 0, False

    if n_votes < MIN_STRONG_VOTES:
        print(
            f"  WARNING: only {n_votes} strong votes (need {MIN_STRONG_VOTES}). "
            f"Global shift declared UNKNOWN — will rely on per-plot imagery alignment."
        )
        return 0.0, 0.0, 0, 0, False

    conf_arr = np.array(conf_votes)
    dx_arr   = np.array(dx_votes)
    dy_arr   = np.array(dy_votes)

    def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        order    = np.argsort(values)
        cum_w    = np.cumsum(weights[order])
        ix       = np.searchsorted(cum_w, cum_w[-1] * 0.5)
        return float(values[order[ix]])

    global_dx = weighted_median(dx_arr, conf_arr)
    global_dy = weighted_median(dy_arr, conf_arr)
    gdx_pix   = int(round(global_dx / pixel_size_x))
    gdy_pix   = int(round(global_dy / pixel_size_y))

    print(
        f"  Global shift accepted: dx={global_dx:.2f} m ({gdx_pix} px)  "
        f"dy={global_dy:.2f} m ({gdy_pix} px)  "
        f"(conf mean={conf_arr.mean():.1f}  dx_std={dx_arr.std():.2f} m)"
    )
    return global_dx, global_dy, gdx_pix, gdy_pix, True


# ---------------------------------------------------------------------------
# Stage 2 — Rubbersheeting (STRONG + WEAK plots only)
# ---------------------------------------------------------------------------

def extract_reliable_local_shifts(
    village,
    bnd_src,
    img_src,
    global_dx_pix: int,
    global_dy_pix: int,
    pixel_size_x:  float,
    pixel_size_y:  float,
    search_radius: int = RUBBER_SEARCH,
    is_low_density: bool = False,
) -> list[dict]:
    """
    Extract per-plot best-fit local shifts from STRONG and WEAK plots.
    Applies saliency + strict local-maximum quality checks on the fused grid.
    """
    step    = max(1, len(village.plots) // RUBBER_N_SAMPLE)
    sampled = village.plots.iloc[::step]
    S       = search_radius
    reliable: list[dict] = []

    for _, row in sampled.iterrows():
        try:
            plot_geom   = row.geometry
            plot_geom_u = geom_to_imagery_crs(bnd_src, plot_geom)

            patch = patch_for_plot_single_band(bnd_src, plot_geom, pad_m=30.0)
            d = hint_density(patch)
            regime = signal_regime(d)
            if regime == "absent":
                continue   # absent plots contribute nothing reliable here

            # ---- Boundary score grid (for strong / weak) ----
            bnd_grid: np.ndarray | None = None
            if regime in ("strong", "weak"):
                hint_m   = (patch.image[:, :, 0] > 0).astype(np.uint8)
                edge_b   = boundary_mask(polygon_mask(plot_geom_u, patch))
                base_sum = edge_b.sum()

                if (base_sum >= 80 and hint_m.sum() > 0 and
                        patch.image.shape[0] > 2 * S and
                        patch.image.shape[1] > 2 * S):
                    bnd_grid = np.zeros((2 * S + 1, 2 * S + 1))
                    for dy_off in range(-S, S + 1):
                        for dx_off in range(-S, S + 1):
                            shifted = shift_mask(
                                edge_b,
                                global_dx_pix + dx_off,
                                -(global_dy_pix + dy_off),
                            )
                            bnd_grid[dy_off + S, dx_off + S] = \
                                overlap_score(shifted, hint_m, base_sum)

            # ---- RGB gradient score grid (for weak, or strong if low density) ----
            rgb_grid: np.ndarray | None = None
            if regime == "weak" or (regime == "strong" and is_low_density):
                rgb_grid = _rgb_gradient_score_grid(
                    img_src, plot_geom, plot_geom_u,
                    global_dx_pix, global_dy_pix, S,
                )

            # ---- Fuse grids ----
            fused = _fuse_grids(bnd_grid, rgb_grid, regime, is_low_density)
            if fused is None:
                continue

            best_idx   = np.unravel_index(np.argmax(fused), fused.shape)
            best_score = fused[best_idx]
            r_c, c_c   = best_idx
            dy_off_b   = r_c - S
            dx_off_b   = c_c - S

            # Reject boundary-hitting peaks
            if abs(dx_off_b) >= S or abs(dy_off_b) >= S:
                continue

            # Strict 3×3 local maximum
            nbhd = fused[r_c-1:r_c+2, c_c-1:c_c+2]
            if best_score < np.max(nbhd) or (nbhd == best_score).sum() > 1:
                continue

            # Saliency: drop vs second-best outside 2-px radius
            second_best = max(
                (fused[r, c]
                 for r in range(fused.shape[0])
                 for c in range(fused.shape[1])
                 if np.sqrt((r - r_c) ** 2 + (c - c_c) ** 2) > 2),
                default=0.0,
            )

            if base_sum >= 80 and best_score > 0.08 and (best_score - second_best) > 0.010:
                dx_m = (global_dx_pix + dx_off_b) * pixel_size_x
                dy_m = (global_dy_pix + dy_off_b) * pixel_size_y
                centroid = plot_geom_u.centroid
                reliable.append({
                    "centroid": (centroid.x, centroid.y),
                    "dx": dx_m,
                    "dy": dy_m,
                    "score": best_score,
                })

        except Exception:
            continue

    return reliable


# ---------------------------------------------------------------------------
# RGB gradient score grid  (used for WEAK and ABSENT plots)
# ---------------------------------------------------------------------------

def _rgb_gradient_score_grid(
    img_src,
    plot_geom,
    plot_geom_u,           # already-projected geometry (avoids re-projecting)
    local_dx_pix: int,
    local_dy_pix: int,
    search_pixels: int,
) -> np.ndarray | None:
    """
    Build a (2S+1)×(2S+1) score grid using Canny-style gradient energy from
    the RGB satellite patch as the hint surface (instead of boundary.tif).

    Uses Sobel gradient magnitude thresholded at the 75th percentile.
    Returns None when the patch is too small or the image is textureless.
    """
    try:
        rgb_patch = patch_for_plot(img_src, plot_geom, pad_m=25.0)
        H, W      = rgb_patch.image.shape[0], rgb_patch.image.shape[1]

        if H <= 2 * search_pixels or W <= 2 * search_pixels:
            return None
        if rgb_patch.image.shape[2] < 3:
            return None

        # Luminance (float 0–1)
        r, g, b = (
            rgb_patch.image[:, :, i].astype(np.float32) / 255.0
            for i in range(3)
        )
        lum  = 0.299 * r + 0.587 * g + 0.114 * b

        # Sobel gradient magnitude
        gx   = scipy_sobel(lum, axis=1)
        gy   = scipy_sobel(lum, axis=0)
        grad = np.hypot(gx, gy)

        thresh = np.percentile(grad, 75)
        if thresh < 1e-6:
            return None   # completely flat image, no edges

        grad_mask = (grad >= thresh).astype(np.uint8)

        # Rasterize polygon edges into the RGB patch coordinate system
        # (rgb_patch may have a different transform/origin from bnd_src patches)
        rgb_geom_u = geom_to_imagery_crs(img_src, plot_geom)
        edge_base  = boundary_mask(polygon_mask(rgb_geom_u, rgb_patch))
        base_sum   = edge_base.sum()
        if base_sum == 0:
            return None

        S = search_pixels
        score_grid = np.zeros((2 * S + 1, 2 * S + 1))
        for dy_off in range(-S, S + 1):
            for dx_off in range(-S, S + 1):
                shifted = shift_mask(edge_base, local_dx_pix + dx_off,
                                               -(local_dy_pix + dy_off))
                score_grid[dy_off + S, dx_off + S] = \
                    overlap_score(shifted, grad_mask, base_sum)
        return score_grid

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Score grid fusion — regime-aware weights
# ---------------------------------------------------------------------------

def _fuse_grids(
    bnd_grid:  np.ndarray | None,
    rgb_grid:  np.ndarray | None,
    regime:    str,
    is_low_density: bool = False,
) -> np.ndarray | None:
    """
    Fuse boundary and RGB score grids according to the signal regime.

      strong : use bnd_grid entirely, unless low density where we anchor with 0.3 * rgb
      weak   : blend (0.5 * each if low density, 0.6 * bnd + 0.4 * rgb otherwise)
      absent : use rgb_grid entirely (bnd is noise)

    Returns None if neither grid is available.
    """
    if regime == "strong":
        if is_low_density and bnd_grid is not None and rgb_grid is not None:
            return 0.7 * bnd_grid + 0.3 * rgb_grid
        return bnd_grid if bnd_grid is not None else rgb_grid

    if regime == "absent":
        return rgb_grid if rgb_grid is not None else bnd_grid

    # weak: blend
    if bnd_grid is not None and rgb_grid is not None:
        w_bnd = 0.5 if is_low_density else 0.6
        w_rgb = 0.5 if is_low_density else 0.4
        return w_bnd * bnd_grid + w_rgb * rgb_grid
    return bnd_grid if bnd_grid is not None else rgb_grid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(village_dir: str = "data/Vadnerbhairav") -> gpd.GeoDataFrame:
    village     = load(village_dir)
    predictions = []
    to_4326     = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # ------------------------------------------------------------------ #
    # No boundary raster at all — output uncorrected with flag            #
    # ------------------------------------------------------------------ #
    if village.boundaries_path is None:
        print("Warning: boundaries.tif not found — outputting uncorrected geometries")
        for _, row in village.plots.iterrows():
            predictions.append(_make_pred(
                row, row.geometry, "flagged", 0.0,
                "no boundaries raster found"
            ))
        return _write_and_return(predictions, village, village_dir)

    # ------------------------------------------------------------------ #
    # Pixel size (read once from the first plot patch)                    #
    # ------------------------------------------------------------------ #
    with open_imagery(village.boundaries_path) as bnd_src:
        _p = patch_for_plot_single_band(
            bnd_src, village.plots.geometry.iloc[0], pad_m=5.0
        )
        pixel_size_x: float = _p.transform.a
        pixel_size_y: float = abs(_p.transform.e)

    # ================================================================== #
    # Stage 0 — Signal census                                             #
    # ================================================================== #
    print("Stage 0: village signal census …")
    with open_imagery(village.boundaries_path) as bnd_src:
        census = village_signal_census(village, bnd_src)

    # ================================================================== #
    # Stage 1 — Global shift (strong plots only)                          #
    # ================================================================== #
    print("Stage 1: global shift estimation …")
    with open_imagery(village.boundaries_path) as bnd_src:
        global_dx, global_dy, global_dx_pix, global_dy_pix, global_valid = \
            estimate_global_shift(village, bnd_src, census,
                                  pixel_size_x, pixel_size_y)

    # ================================================================== #
    # Stage 2 — Rubbersheeting                                            #
    # ================================================================== #
    reliable_samples: list[dict] = []
    is_low_density = (census["mean_density"] < 0.04)

    print("Stage 2: extracting local reference shifts …")
    t0 = time.time()
    with open_imagery(village.boundaries_path) as bnd_src, \
         open_imagery(village.imagery_path) as img_src:
        start_dx = global_dx_pix if global_valid else 0
        start_dy = global_dy_pix if global_valid else 0
        s_radius = RUBBER_SEARCH if global_valid else 8  # 8 pixels = ~19 meters
        
        reliable_samples = extract_reliable_local_shifts(
            village, bnd_src, img_src,
            start_dx, start_dy,
            pixel_size_x, pixel_size_y,
            search_radius=s_radius,
            is_low_density=is_low_density,
        )
    print(
        f"  {len(reliable_samples)} reliable reference shifts "
        f"in {time.time() - t0:.2f} s"
    )
    print(f"[Stage 2 Shifts] {[ (round(s['dx'], 2), round(s['dy'], 2), round(s['score'], 3)) for s in reliable_samples ]}")

    sample_coords = (
        np.array([s["centroid"] for s in reliable_samples])
        if reliable_samples else np.array([])
    )

    # ================================================================== #
    # Stage 3 — Per-plot fine alignment                                   #
    # ================================================================== #
    print("Stage 3: per-plot fine alignment …")
    S  = FINE_SEARCH
    t0 = time.time()

    with open_imagery(village.imagery_path)   as img_src, \
         open_imagery(village.boundaries_path) as bnd_src:

        for _, row in village.plots.iterrows():
            try:
                plot_geom   = row.geometry
                plot_geom_u = geom_to_imagery_crs(bnd_src, plot_geom)
                centroid    = plot_geom_u.centroid

                # ---- Determine per-plot signal regime ----
                bnd_patch = patch_for_plot_single_band(bnd_src, plot_geom, pad_m=25.0)
                d         = hint_density(bnd_patch)
                regime    = signal_regime(d)

                # ---- KNN local shift (or global fallback or zero) ----
                if len(reliable_samples) > 0:
                    dists = np.sqrt(np.sum(
                        (sample_coords - np.array([centroid.x, centroid.y])) ** 2,
                        axis=1,
                    ))
                    valid_idx = np.where(dists < 200.0)[0]
                    if len(valid_idx) > 0:
                        sorted_valid = valid_idx[np.argsort(dists[valid_idx])]
                        nn_idx = sorted_valid[:3]
                        local_dx = float(np.median([reliable_samples[i]["dx"] for i in nn_idx]))
                        local_dy = float(np.median([reliable_samples[i]["dy"] for i in nn_idx]))
                    else:
                        local_dx, local_dy = (global_dx, global_dy) if global_valid else (0.0, 0.0)
                elif global_valid:
                    local_dx, local_dy = global_dx, global_dy
                else:
                    local_dx, local_dy = 0.0, 0.0

                local_dx_pix = int(round(local_dx / pixel_size_x))
                local_dy_pix = int(round(local_dy / pixel_size_y))

                # ---- Boundary score grid (for strong / weak) ----
                bnd_grid: np.ndarray | None = None
                if regime in ("strong", "weak"):
                    hint_m   = (bnd_patch.image[:, :, 0] > 0).astype(np.uint8)
                    edge_b   = boundary_mask(polygon_mask(plot_geom_u, bnd_patch))
                    base_sum = edge_b.sum()

                    if (base_sum > 0 and hint_m.sum() > 0 and
                            bnd_patch.image.shape[0] > 2 * S and
                            bnd_patch.image.shape[1] > 2 * S):
                        bnd_grid = np.zeros((2 * S + 1, 2 * S + 1))
                        for dy_off in range(-S, S + 1):
                            for dx_off in range(-S, S + 1):
                                shifted = shift_mask(
                                    edge_b,
                                    local_dx_pix + dx_off,
                                    -(local_dy_pix + dy_off),
                                )
                                bnd_grid[dy_off + S, dx_off + S] = \
                                    overlap_score(shifted, hint_m, base_sum)

                # ---- RGB gradient score grid (for weak / absent, or strong if low density) ----
                rgb_grid: np.ndarray | None = None
                if regime in ("weak", "absent") or (regime == "strong" and is_low_density):
                    rgb_grid = _rgb_gradient_score_grid(
                        img_src, plot_geom, plot_geom_u,
                        local_dx_pix, local_dy_pix, S,
                    )

                # ---- Fuse grids ----
                fused = _fuse_grids(bnd_grid, rgb_grid, regime, is_low_density)

                if fused is None:
                    # No signal at all — flag and keep original geometry
                    predictions.append(_make_pred(
                        row,
                        row.geometry,
                        "flagged", 0.0,
                        f"no signal (regime={regime}) knn dx={local_dx:.2f} dy={local_dy:.2f}",
                    ))
                    continue

                # ---- Peak extraction ----
                best_idx    = np.unravel_index(np.argmax(fused), fused.shape)
                best_score  = fused[best_idx]
                best_dy_off = best_idx[0] - S
                best_dx_off = best_idx[1] - S

                flat_sorted = np.sort(fused.flatten())
                second_best = flat_sorted[-2] if len(flat_sorted) > 1 else 0.0
                drop        = best_score - second_best

                ref_dx = local_dx + best_dx_off * pixel_size_x
                ref_dy = local_dy + best_dy_off * pixel_size_y

                # Confidence threshold is slightly relaxed for ABSENT plots
                # because rgb_grid scores tend to be lower than boundary scores
                min_score = 0.04 if regime != "absent" else 0.02
                min_drop  = 0.003

                is_confident = (best_score >= min_score) and (drop >= min_drop)

                if is_confident:
                    val = (best_score / (d + 0.01)) + 20.0 * drop
                    if regime == "weak":
                        val *= 0.8
                    elif regime == "absent":
                        val *= 0.6
                    confidence   = min(0.95, max(0.1, 0.18 * val + 0.05))
                    corrected_u  = translate(plot_geom_u, ref_dx, ref_dy)
                    note = (
                        f"regime={regime} "
                        f"dx={ref_dx:.2f} dy={ref_dy:.2f} "
                        f"score={best_score:.3f} drop={drop:.3f}"
                    )
                    predictions.append(_make_pred(
                        row,
                        shp_transform(to_4326.transform, corrected_u),
                        "corrected", confidence, note,
                    ))
                else:
                    confidence  = best_score * 0.5
                    note = (
                        f"regime={regime} weak peak "
                        f"knn dx={local_dx:.2f} dy={local_dy:.2f}"
                    )
                    predictions.append(_make_pred(
                        row,
                        row.geometry,
                        "flagged", confidence, note,
                    ))

                print(
                    regime,
                    confidence,
                    best_score,
                    drop
                )

            except Exception as e:
                predictions.append(_make_pred(
                    row, row.geometry, "flagged", 0.0, f"Error: {e}"
                ))

    print(f"  All plots processed in {time.time() - t0:.2f} s")
    return _write_and_return(predictions, village, village_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pred(row, geometry, status, confidence, method_note) -> dict:
    return {
        "plot_number": row.plot_number,
        "status":      status,
        "confidence":  float(confidence),
        "method_note": method_note,
        "geometry":    geometry,
    }


def _write_and_return(
    predictions: list[dict],
    village,
    village_dir: str,
) -> gpd.GeoDataFrame:
    pred_gdf = gpd.GeoDataFrame(predictions, geometry="geometry", crs="EPSG:4326")
    pred_gdf["plot_number"] = pred_gdf["plot_number"].astype(str)
    pred_gdf = pred_gdf.set_index("plot_number", drop=False)

    out_path = Path(village_dir) / "predictions.geojson"
    write_predictions(out_path, pred_gdf)
    print(f"Predictions written to: {out_path}")

    if village.example_truths is not None:
        from bhume import score
        print()
        print(score(pred_gdf, village))

    return pred_gdf


if __name__ == "__main__":
    import sys
    village_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Vadnerbhairav"
    run(village_dir)