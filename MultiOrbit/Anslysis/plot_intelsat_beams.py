import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import simpy

from LEOEnvironmentRL import create_Constellation

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


def _ellipse_outline(center_lon, center_lat, width_deg, height_deg, n=120):
    a = width_deg / 2.0
    b = height_deg / 2.0
    t = np.linspace(0.0, 2.0 * math.pi, n)
    xs = center_lon + a * np.cos(t)
    ys = center_lat + b * np.sin(t)
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description="Plot Intelsat GEO satellite beams.")
    parser.add_argument(
        "--constellation",
        default="Intelsat",
        help="Constellation name to instantiate (default: Intelsat).",
    )
    parser.add_argument(
        "--output",
        default="intelsat_beams.png",
        help="Output image path (default: intelsat_beams.png).",
    )
    parser.add_argument(
        "--na-view",
        action="store_true",
        help="Use North America focused extent instead of global view.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving the image.",
    )
    args = parser.parse_args()

    env = simpy.Environment()
    orbital_planes = create_Constellation(args.constellation, env)
    if not orbital_planes:
        raise RuntimeError("No orbital planes created.")

    sats = orbital_planes[0].sats
    if not sats:
        raise RuntimeError("No satellites created.")

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title(f"{args.constellation} GEO Beams")
        if args.na_view:
            # Optional focus where the synthetic Intelsat beams are defined.
            ax.set_extent([-170, -40, 5, 75], crs=ccrs.PlateCarree())
        else:
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor="#f2f2f2", alpha=0.8)
        ax.add_feature(cfeature.OCEAN, facecolor="#dceeff", alpha=0.6)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
    else:
        print("Cartopy not available; using plain lon/lat axes without basemap.")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title(f"{args.constellation} GEO Beams (no map backend)")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, alpha=0.25)

    base_cmap = plt.colormaps["tab10"]
    sat_colors = [base_cmap(i % base_cmap.N) for i in range(len(sats))]
    legend_handles = []

    for sat_idx, sat in enumerate(sats):
        sat_lon = math.degrees(sat.longitude)
        sat_lat = math.degrees(sat.latitude)
        color = sat_colors[sat_idx]

        # Satellite marker
        ax.scatter(
            [sat_lon],
            [sat_lat],
            color=color,
            marker="x",
            s=80,
            zorder=5,
            transform=ccrs.PlateCarree() if HAS_CARTOPY else None,
        )
        ax.text(
            sat_lon + 1.5,
            sat_lat + 1.5,
            f"{sat.ID} ({sat_lon:.1f}°)",
            fontsize=8,
            color=color,
            transform=ccrs.PlateCarree() if HAS_CARTOPY else None,
        )

        # Beam outlines
        for beam in sat.beams:
            xs, ys = _ellipse_outline(
                beam.center_lon,
                beam.center_lat,
                beam.width_deg,
                beam.height_deg,
            )
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.0,
                alpha=0.75,
                transform=ccrs.PlateCarree() if HAS_CARTOPY else None,
            )

        # Proxy handle for legend
        handle = plt.Line2D([0], [0], color=color, lw=2, label=f"{sat.ID}: {len(sat.beams)} beams")
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    print(f"Saved beam plot to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
