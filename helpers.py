"""Functions and code used for the IIW presentation notebook"""


import matplotlib.pyplot as plt
import numpy as np

import weldx
from weldx import Q_, SpatialData
from weldx.geometry import Geometry, LinearHorizontalTraceSegment, Trace

_DEFAUL_FIGWIDTH = 10

cs_colors = {
    "workpiece": (100, 100, 100),
    "workpiece geometry": (100, 100, 100),
    "scan_0": (100, 100, 100),
    "scan_1": (100, 100, 100),
    "workpiece geometry (reduced)": (0, 0, 0),
    "workpiece (simple)": (0, 0, 0),
    "user_frame": (180, 180, 0),
    "TCP": (255, 0, 0),
    "TCP design": (200, 0, 0),
    "T1": (0, 255, 0),
    "T2": (0, 200, 0),
    "T3": (0, 150, 0),
    "T4": (0, 100, 0),
    "welding_wire": (150, 150, 0),
    "flange": (0, 0, 255),
    "LLT_1": (40, 240, 180),
    "LLT_2": (20, 190, 150),
    "XIRIS_1": (255, 0, 255),
    "XIRIS_2": (200, 0, 200),
}


def welding_wire_geo_data(radius, length, cross_section_resolution=8):
    points = []
    triangles = []
    for i in range(cross_section_resolution):
        angle = i / cross_section_resolution * np.pi * 2
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        points.append([x, y, 0])
        points.append([x, y, length])

        idx = 2 * i
        triangles.append([idx, idx + 1, idx + 3])
        triangles.append([idx, idx + 3, idx + 2])

    triangles[-2][2] = 1
    triangles[-1][1] = 1
    triangles[-1][2] = 0

    return SpatialData(
        np.array(points, dtype="float32"), np.array(triangles, dtype="uint32")
    )


def plot_signal(signal, name, ref_time=None, limits=None, ax=None):
    """Plot a single weldx signal."""
    if not ax:
        _, ax = plt.subplots(figsize=(_DEFAUL_FIGWIDTH, 6))

    data = signal.data
    time = weldx.util.pandas_time_delta_to_quantity(data.time)

    ax.plot(time.m, data.data)
    ax.set_ylabel(f"{name} / {signal.unit}")
    ax.set_xlabel("time / s")
    ax.grid()

    if limits is not None:
        ax.set_xlim(limits)

    ipympl_style(ax.figure)


def plot_measurements(measurement_data, limits=None, ref_time=None):
    n = len(measurement_data)
    fig, ax = plt.subplots(nrows=n, sharex="all", figsize=(_DEFAUL_FIGWIDTH, 2.5 * n))

    for i, measurement in enumerate(measurement_data):
        last_signal = measurement.measurement_chain.signals[-1]
        plot_signal(
            last_signal, measurement.name, ax=ax[i], limits=limits, ref_time=ref_time
        )
        ax[i].set_xlabel(None)

    ax[-1].set_xlabel("time / s")
    ax[0].set_title("Measurements")

    ipympl_style(fig)


def parplot(par, t, name, ax):
    """plot a single parameter into an axis"""
    ts = par.interp_time(t)
    x = weldx.util.pandas_time_delta_to_quantity(t)
    ax.plot(x.m, ts.data.m)
    ax.set_ylabel(f"{name} / {ts.data.u:~}")
    ax.grid()


def ipympl_style(fig, toolbar=True):
    """Apply default figure styling for ipympl backend."""

    try:
        fig.canvas.header_visible = False
        fig.canvas.resizable = False
        fig.tight_layout()
        fig.canvas.toolbar_position = "right"
        fig.canvas.toolbar_visible = toolbar
    except Exception as ex:
        pass


def plot_gmaw(gmaw, t):
    """Plot a dictionary of parameters"""

    title = "\n".join([gmaw.manufacturer, gmaw.power_source, gmaw.base_process])

    pars = gmaw.parameters
    n = len(pars)

    fig, ax = plt.subplots(nrows=n, sharex="all", figsize=(_DEFAUL_FIGWIDTH, 2 * n))
    for i, k in enumerate(pars):
        parplot(pars[k], t, k, ax[i])
    ax[-1].set_xlabel(f"time / s")
    ax[0].set_title(title, loc="left")

    ipympl_style(fig)

    return fig, ax


def create_geometry(groove, seam_length, width):
    trace = Trace(LinearHorizontalTraceSegment(seam_length))
    return Geometry(groove.to_profile(width_default=width), trace)


def ax_setup(ax):
    # ax.legend()
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.set_zlabel("z / mm")
    ax.view_init(30, -10)
    ax.set_ylim([-10.5, 10.5])
    ax.set_zlim([0, 15])
    ax.figure.set_size_inches(8, 8)
    ipympl_style(ax.figure)


def add_axis_labels_3d(axes):
    axes.set_xlabel("x / mm")
    axes.set_ylabel("y / mm")
    axes.set_zlabel("z / mm")


def build_base_csm(weldx_file: dict, plot=True):
    """Create a simple CSM instance from workpiece information and the TCP movement."""
    seam_length = weldx_file["workpiece"]["geometry"]["seam_length"]
    groove = weldx_file["workpiece"]["geometry"]["groove_shape"]
    geometry = create_geometry(groove, seam_length, Q_(10, "mm"))

    csm = weldx.CoordinateSystemManager("workpiece")
    csm.add_cs("TCP weld", "workpiece", lcs=weldx_file["TCP"])

    spatial_data_geo_reduced = geometry.spatial_data(
        profile_raster_width=Q_(4, "mm"), trace_raster_width=Q_(60, "mm")
    )

    csm.assign_data(spatial_data_geo_reduced, "workpiece (simple)", "workpiece")

    if plot:
        csm.plot(
            reference_system="workpiece",
            coordinate_systems=["TCP weld"],
            data_sets=["workpiece (simple)"],
            colors=cs_colors,
            show_wireframe=True,
            show_data_labels=False,
            show_vectors=False,
        )
        ax_setup(plt.gca())
    return csm
