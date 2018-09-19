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
                 ylimsp=(25, 55),
                 ylimsm=(25, 55),
                 percp=15,
                 percm=15):
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
        :param ylimsp: Meridional limits between which northward
        current is checked
        :param ylimsm: Meridional limits between which southward
        current is checked
        :param percp: Percentile for northward current
        :param percm: Percentile for southward current
        :returns: Experiment class
        :rtype: Experiment

        """
        self.name = name
        self.path = path
        self.fil0 = [
            self.path + f'output__{n:04}.nc'
            for n in range(start_year, end_year + 1)
        ]
        self.fil1 = [
            self.path + f'twaoutput__{n:04}.nc'
            for n in range(start_year, end_year + 1)
        ]
        self.fil2 = [
            self.path + f'avg_output__{n:04}.nc'
            for n in range(start_year, end_year + 1)
        ]
        self.fil3 = self.path + f'twainstoutput__{end_year:04}.nc'
        self.geometry = pym6.GridGeometry(self.path + './ocean_geometry.nc')
        self.z1 = z1
        self.z2 = z2
        self.ylimsp = ylimsp
        self.ylimsm = ylimsm
        self.percp = percp
        self.percm = percm


class ExperimentsList():
    """Holds a list of Experiments."""

    def __init__(self, *args):
        self.list_ = args

    def plot2d(self,
               func,
               *func_args,
               contourf=False,
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
                im = arg.plot.contourf(ax=axc, **plot_kwargs)
            else:
                im = arg.plot(ax=axc, **plot_kwargs)
            if contours:
                arg.squeeze().plot.contour(
                    ax=axc, add_labels=False, **ctr_kwargs)
            axc.set_title(exp.name)
        if plot_kwargs.get('add_colorbar', True) is False:
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
