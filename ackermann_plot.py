"""
Ackermann steering geometry calculator and plotter.

Given:
  - steering input  delta  (steering angle, radians)
  - wheelbase       L      (metres)
  - track width     T      (metres)

Computes:
  R       = L / tan(delta)               (turning radius at rear-axle centre)
  R_min   = sqrt((R + T/2)^2 + L^2)     (minimum turning radius – outer front corner)
  delta_i = arctan(L / (R - T/2))        (inner wheel angle)
  delta_o = arctan(L / (R + T/2))        (outer wheel angle)

Pulley-system analysis (assuming phi = delta as the steering-wheel angle):
  n_i = phi / delta_i  =  r_i / R_sw    (inner-wheel angle ratio = normalised inner pulley radius)
  n_o = phi / delta_o  =  r_o / R_sw    (outer-wheel angle ratio = normalised outer pulley radius)

Produces a PNG plot of R, R_min, delta_i, and delta_o vs delta,
and a second PNG plot of the pulley ratios and normalised radii vs delta.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # non-interactive backend for PNG output


def compute_ackermann(delta: np.ndarray, L: float, T: float):
    """Return (R, R_min, delta_i, delta_o) for arrays of steering angle *delta* (rad).

    R       – turning radius at the rear-axle centre  (= L / tan(delta))
    R_min   – minimum turning radius of the vehicle body, swept by the outer
              front corner: sqrt((R + T/2)^2 + L^2).  This is the quantity that
              fully accounts for both the wheelbase *L* and the track width *T*.
    delta_i – inner front-wheel angle
    delta_o – outer front-wheel angle

    Entries where ``delta == 0`` (straight-ahead, undefined R) or where
    ``R <= T/2`` (geometrically impossible inner-wheel pivot) are returned as
    ``np.nan`` so the caller can handle or mask them as appropriate.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.tan(delta) != 0, L / np.tan(delta), np.nan)
        denom_i = R - T / 2.0
        denom_o = R + T / 2.0
        delta_i = np.where(denom_i > 0, np.arctan(L / denom_i), np.nan)
        delta_o = np.where(denom_o > 0, np.arctan(L / denom_o), np.nan)
        # Minimum turning radius: distance from turn centre to outer front corner
        R_min = np.where(np.isfinite(R), np.sqrt((R + T / 2.0) ** 2 + L ** 2), np.nan)
    return R, R_min, delta_i, delta_o


def plot_ackermann(
    L: float = 2.5,
    T: float = 1.5,
    delta_min_deg: float = 20.0,
    delta_max_deg: float = 90.0,
    output_file: str = "ackermann_plot.png",
) -> str:
    """
    Plot Ackermann steering geometry functions and save to *output_file*.

    Parameters
    ----------
    L : float
        Wheelbase in metres.
    T : float
        Track width in metres.
    delta_min_deg : float
        Minimum steering angle in degrees (must be > 0).
    delta_max_deg : float
        Maximum steering angle in degrees.
    output_file : str
        Path of the output PNG file.

    Returns
    -------
    str
        Path of the saved PNG file.
    """
    if delta_min_deg <= 0:
        raise ValueError("delta_min_deg must be greater than 0 to avoid division by zero.")
    if delta_max_deg <= delta_min_deg:
        raise ValueError("delta_max_deg must be greater than delta_min_deg.")

    delta_deg = np.linspace(delta_min_deg, delta_max_deg, 500)
    delta_rad = np.deg2rad(delta_deg)

    R, R_min, delta_i, delta_o = compute_ackermann(delta_rad, L, T)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(
        f"Ackermann Steering Geometry\n(L = {L} m, T = {T} m)",
        fontsize=14,
    )

    # --- top panel: turning radius ---
    ax_r = axes[0]
    ax_r.plot(delta_deg, R, color="tab:blue", linewidth=2, label=r"$R$ (rear axle)")
    ax_r.plot(
        delta_deg,
        R_min,
        color="tab:red",
        linewidth=2,
        linestyle="--",
        label=r"$R_{\mathrm{min}}$ (outer front corner)",
    )
    ax_r.set_ylabel("Turning radius (m)", fontsize=11)
    ax_r.set_title(
        r"$R = \dfrac{L}{\tan(\delta)}$"
        r"  ,  "
        r"$R_{\mathrm{min}} = \sqrt{(R + T/2)^2 + L^2}$",
        fontsize=11,
    )
    ax_r.legend(fontsize=10)
    ax_r.grid(True, linestyle="--", alpha=0.6)

    # --- bottom panel: inner and outer wheel angles ---
    ax_a = axes[1]
    ax_a.plot(
        delta_deg,
        np.rad2deg(delta_i),
        color="tab:orange",
        linewidth=2,
        label=r"$\delta_i$ (inner)",
    )
    ax_a.plot(
        delta_deg,
        np.rad2deg(delta_o),
        color="tab:green",
        linewidth=2,
        linestyle="--",
        label=r"$\delta_o$ (outer)",
    )
    ax_a.plot(
        delta_deg,
        delta_deg,
        color="tab:gray",
        linewidth=1,
        linestyle=":",
        label=r"$\delta$ (input)",
    )
    ax_a.set_xlabel(r"Steering input $\delta$ (degrees)", fontsize=11)
    ax_a.set_ylabel("Wheel angle (degrees)", fontsize=11)
    ax_a.set_title(
        r"$\delta_i = \arctan\!\left(\frac{L}{R - T/2}\right)$"
        r"  ,  "
        r"$\delta_o = \arctan\!\left(\frac{L}{R + T/2}\right)$",
        fontsize=11,
    )
    ax_a.legend(fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file


def compute_pulley_ratios(delta: np.ndarray, L: float, T: float):
    """Return normalised pulley radii (ratio_i, ratio_o) for steering angle arrays *delta* (rad).

    Assuming a pulley system where the steering-wheel pulley (radius R_sw) drives
    one pulley per wheel via an inextensible belt, arc-length conservation gives:

        R_sw * phi = r_wheel * delta_wheel   =>   r_wheel / R_sw = phi / delta_wheel

    Taking the steering-wheel angle phi = delta (the Ackermann reference angle):

        n_i = delta / delta_i   (= r_i / R_sw, the normalised inner-wheel pulley radius)
        n_o = delta / delta_o   (= r_o / R_sw, the normalised outer-wheel pulley radius)

    Values where the geometry is undefined are returned as ``np.nan``.
    """
    _, _, delta_i, delta_o = compute_ackermann(delta, L, T)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_i = np.where((delta_i > 0) & np.isfinite(delta_i), delta / delta_i, np.nan)
        n_o = np.where((delta_o > 0) & np.isfinite(delta_o), delta / delta_o, np.nan)
    return n_i, n_o


def plot_pulley_ratios(
    L: float = 2.5,
    T: float = 1.5,
    delta_min_deg: float = 20.0,
    delta_max_deg: float = 90.0,
    output_file: str = "pulley_plot.png",
) -> str:
    """
    Plot the steering-wheel-to-wheel angle ratios and the corresponding
    normalised pulley radii for a pulley-based Ackermann steering system,
    and save to *output_file*.

    For an inextensible-belt pulley system the wheel-pulley radius required
    to achieve perfect Ackermann geometry is:

        r_i / R_sw = phi / delta_i = delta / arctan(L / (L/tan(delta) - T/2))
        r_o / R_sw = phi / delta_o = delta / arctan(L / (L/tan(delta) + T/2))

    where phi = delta is taken as the steering-wheel (reference) angle.

    Parameters
    ----------
    L : float
        Wheelbase in metres.
    T : float
        Track width in metres.
    delta_min_deg : float
        Minimum steering angle in degrees (must be > 0).
    delta_max_deg : float
        Maximum steering angle in degrees.
    output_file : str
        Path of the output PNG file.

    Returns
    -------
    str
        Path of the saved PNG file.
    """
    if delta_min_deg <= 0:
        raise ValueError("delta_min_deg must be greater than 0 to avoid division by zero.")
    if delta_max_deg <= delta_min_deg:
        raise ValueError("delta_max_deg must be greater than delta_min_deg.")

    delta_deg = np.linspace(delta_min_deg, delta_max_deg, 500)
    delta_rad = np.deg2rad(delta_deg)

    n_i, n_o = compute_pulley_ratios(delta_rad, L, T)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle(
        f"Pulley-System Steering Ratios (Ackermann)\n(L = {L} m, T = {T} m)",
        fontsize=14,
    )

    # --- top panel: steering-wheel-to-wheel angle ratio ---
    ax_ratio = axes[0]
    ax_ratio.plot(delta_deg, n_i, color="tab:orange", linewidth=2,
                  label=r"$n_i = \varphi / \delta_i$ (inner)")
    ax_ratio.plot(delta_deg, n_o, color="tab:green", linewidth=2, linestyle="--",
                  label=r"$n_o = \varphi / \delta_o$ (outer)")
    ax_ratio.axhline(1.0, color="tab:gray", linewidth=1, linestyle=":", label="ratio = 1")
    ax_ratio.set_ylabel("Angle ratio  " r"$\varphi / \delta_{\mathrm{wheel}}$", fontsize=11)
    ax_ratio.set_title(
        r"$n_i = \dfrac{\varphi}{\delta_i}$"
        r"  ,  "
        r"$n_o = \dfrac{\varphi}{\delta_o}$"
        r"  (with $\varphi = \delta$)",
        fontsize=11,
    )
    ax_ratio.legend(fontsize=10)
    ax_ratio.grid(True, linestyle="--", alpha=0.6)

    # --- bottom panel: normalised pulley radius (= n_i, n_o by arc-length conservation) ---
    ax_r = axes[1]
    ax_r.plot(delta_deg, n_i, color="tab:orange", linewidth=2,
              label=r"$r_i / R_{\mathrm{sw}}$ (inner)")
    ax_r.plot(delta_deg, n_o, color="tab:green", linewidth=2, linestyle="--",
              label=r"$r_o / R_{\mathrm{sw}}$ (outer)")
    ax_r.axhline(1.0, color="tab:gray", linewidth=1, linestyle=":")
    ax_r.set_xlabel(r"Steering input $\delta$ (degrees)", fontsize=11)
    ax_r.set_ylabel(r"Normalised pulley radius  $r / R_{\mathrm{sw}}$", fontsize=11)

    closed_form = (
        r"$\dfrac{r_i}{R_\mathrm{sw}} = "
        r"\dfrac{\delta}{\arctan\!\left(\dfrac{L}{L/\tan(\delta)\,-\,T/2}\right)}$"
        "\n"
        r"$\dfrac{r_o}{R_\mathrm{sw}} = "
        r"\dfrac{\delta}{\arctan\!\left(\dfrac{L}{L/\tan(\delta)\,+\,T/2}\right)}$"
    )
    ax_r.set_title(closed_form, fontsize=10)
    ax_r.legend(fontsize=10)
    ax_r.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file


if __name__ == "__main__":
    saved = plot_ackermann()
    print(f"Plot saved to: {saved}")
    saved_pulley = plot_pulley_ratios()
    print(f"Pulley plot saved to: {saved_pulley}")
