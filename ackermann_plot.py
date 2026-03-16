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

Produces a PNG plot of R, R_min, delta_i, and delta_o vs delta.
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


if __name__ == "__main__":
    saved = plot_ackermann()
    print(f"Plot saved to: {saved}")
