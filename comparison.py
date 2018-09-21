import numpy as np
from netCDF4 import Dataset as dset, MFDataset as mfdset
from pymom6 import pymom6 as pym6
import importlib
import matplotlib.pyplot as plt
import xarray as xr
import twamomy_budget as ty
import energetics as ee
import twapv_budget as tp
import plot_budget as pb
import scipy.linalg as sl
from scipy.optimize import curve_fit
importlib.reload(pym6)


class Experiment():
    def __init__(self,
                 name,
                 path,
                 start_year,
                 end_year,
                 z1=None,
                 z2=None,
                 yn=50,
                 ylimsp=(25, 55),
                 ylimsm=(25, 55),
                 percp=15,
                 percm=15,
                 dt=10):
        """Defines a MOM6 experiment

        :param name: Name of the experiment. This will appear on the
        figures
        :param path: Path of the folder in which output is located
        :param start_year: year when output was started
        :param end_year: year when output ended
        :param z1: z1 to check depth of current (see description of
        z2)
        :param z2: z2 (Depth of the change of current direction is
        checked between z1 and z2)
        :param yn: Northernmost outcrop location (degrees)
        :param ylimsp: Meridional limits between which northward
        current is checked
        :param ylimsm: Meridional limits between which southward
        current is checked
        :param percp: Percentile for northward current
        :param percm: Percentile for southward current
        :param dt: Imposed temperature gradient at the surface
        :returns: Experiment class
        :rtype: Experiment

        """
        self.name = name
        self.path = path
        self.fil0 = [
            self.path + 'output__{:04}.nc'.format(n)
            for n in range(start_year, end_year + 1)
        ]
        self.fil1 = [
            self.path + 'twaoutput__{:04}.nc'.format(n)
            for n in range(start_year, end_year + 1)
        ]
        self.fil2 = [
            self.path + 'avg_output__{:04}.nc'.format(n)
            for n in range(start_year, end_year + 1)
        ]
        self.fil3 = self.path + 'twainstoutput__{:04}.nc'.format(end_year)
        self.geometry = pym6.GridGeometry(self.path + './ocean_geometry.nc')
        self.z1 = z1
        self.z2 = z2
        self.ylimsp = ylimsp
        self.ylimsm = ylimsm
        self.percp = percp
        self.percm = percm
        self.dt = dt
        omega = 2 * np.pi * (1 / 24 / 3600 + 1 / 24 / 3600 / 365)
        self.fn = 2 * omega * np.sin(np.radians(yn))


class ExperimentsList():
    """Holds a list of Experiments."""

    def __init__(self, *args):
        self.list_ = args

    def plot2d(self,
               func,
               *func_args,
               contourf=True,
               contours=False,
               fig=None,
               plot_kwargs={},
               ctr_kwargs={},
               **func_kwargs):
        """Generates 2D plots from a list of experiments and arranges
               them on a subplot

        :param func: Function that returns a plottable object (xarray object)
        :param contourf: Uses contourf if true
        :param contours: Plots contours if true
        :param fig: Figure handle, creates a new one if None
        :param plot_kwargs: These are passed to the plotting function
        :param ctr_kwargs: These are passed to the contouring function
        :returns: fig
        :rtype: Matplotlib Figure

        """
        if fig is None:
            fig, ax = plt.subplots(
                2, int(np.ceil(len(self.list_) / 2)), sharex=True, sharey=True)
            ax = ax.ravel().tolist()
        else:
            ax = fig.axes
            assert len(ax) >= len(self.list_)
        for exp, axc in zip(self.list_, ax[:len(self.list_)]):
            arg = func(exp, *func_args, **func_kwargs)
            if contourf:
                im = arg.plot.pcolormesh(ax=axc, **plot_kwargs)
            if contours:
                arg.squeeze().plot.contour(
                    ax=axc, add_labels=False, **ctr_kwargs)
            axc.set_title(exp.name)
        if contourf and plot_kwargs.get('add_colorbar', True) is False:
            fig.colorbar(im, ax=ax)
        return fig

    def plot1d(self, func, *func_args, ax=None, plot_kwargs={}, **func_kwargs):
        """Creates a 1D plots on the same axis

        :param func: Function that returns a plottable object
        :param ax: Axis object, new axis is created if None
        :param plot_kwargs: These are passed to the plotting function
        :returns: fig
        :rtype: Matplotlib figure

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        for exp in self.list_:
            func(exp, *func_args, **func_kwargs).plot(
                ax=ax, label=exp.name, **plot_kwargs)
        ax.legend(loc='best')
        return ax.get_figure()

    def plotpoint(self,
                  func,
                  *func_args,
                  ax=None,
                  plot_kwargs={},
                  **func_kwargs):
        """Creates a figure with a point representing each experiment

        :param func: Function that returns a data
        :param ax: Axis object, new axis is created if None
        :param plot_kwargs: These are passed to the plotting function
        :returns: fig
        :rtype: Matplotlib figure

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        x, labels, data = zip(*[(i, exp.name,
                                 func(exp, *func_args, **func_kwargs))
                                for i, exp in enumerate(self.list_)])
        ax.plot(x, data, **plot_kwargs)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid()
        fig = ax.get_figure()
        fig.autofmt_xdate()
        return fig


control = Experiment(
    r'Control',
    './',
    38,
    47,
    yn=50,
    z1=-200,
    z2=-1300,
    ylimsp=(26, 55),
    ylimsm=(25, 52))
decdt = Experiment(
    r'$\Delta$T = 8$^{\circ}$C',
    '../bfb3_fixed_sponge_decdt/',
    66,
    72,
    yn=45,
    z1=-200,
    z2=-1100,
    ylimsp=(28, 50),
    ylimsm=(30, 50),
    dt=8)
incdt = Experiment(
    r'$\Delta$T = 12$^{\circ}$C',
    '../bfb3_fixed_sponge_incdt/',
    55,
    61,
    yn=55,
    z1=-200,
    z2=-1300,
    ylimsp=(25, 55),
    ylimsm=(25, 52),
    percp=2,
    dt=12)
deckd = Experiment(
    'KD = 5e-5', '../bfb3_fixed_sponge_deckd/', 43, 49, z1=-200, z2=-1300)
decdeckd = Experiment(
    'KD = 2.5e-5', '../bfb3_fixed_sponge_decdeckd/', 57, 64, z1=-200, z2=-1300)
incdepth = Experiment(
    'incdepth', '../bfb3_fixed_sponge_lowNsq/', 32, 40, z1=-200, z2=-1300)
decdepth = Experiment(
    'decdepth', '../bfb3_fixed_sponge_highNsq/', 33, 40, z1=-200, z2=-1100)
fplane = Experiment(
    'f-plane', '../bfb3_fixed_sponge_fplane/', 84, 90, z1=-200, z2=-1300)
lowflc = Experiment(
    'Flux Const = 1',
    '../bfb3_fixed_sponge_lowflc/',
    61,
    68,
    z1=-150,
    z2=-1300)
wind1 = Experiment(
    r'$\tau^Y = -0.1$',
    '../bfb3_fixed_sponge_wind_high/',
    118,
    127,
    z1=-200,
    z2=-1400,
    ylimsm=(25, 40))
wind2 = Experiment(
    r'$\tau^Y = -0.5$',
    '../bfb3_fixed_sponge_wind_veryhigh/',
    133,
    142,
    z1=-400,
    z2=-1300,
    ylimsm=(25, 50),
    ylimsp=(25, 50))
