import numpy as np
import xarray as xr
import pymom6.pymom6 as pym6
import matplotlib.pyplot as plt


def rolling_mean(array, n):
    """Returns a rolling mean of a 1D array

    :param array: 1D array
    :param n: Number of points over which rolling mean is taken
    :returns: array
    :rtype: Array

    """
    return np.convolve(array, np.ones((n, )) / n, mode='same')


def get_ntrans(exp, func, N=1, sum_axes=(1, 3), **initializer):
    with pym6.Dataset(exp.fil2, final_loc='vl', **initializer) as ds:
        vh = ds.vh.isel(z=slice(0, -1)).read().nanmean(axis=0).where(
            func, 0, y=0).reduce_(
                np.sum, axis=sum_axes)
        method = initializer.get('method', None)
        if method:
            vh = vh.reduce_(method, axis=2)
        vh.name = r'v (Sv)'
        vh = vh.to_DataArray().squeeze() * 1e-6
        if func == np.less:
            vh = -vh
        if len(vh.shape) == 1:
            vh.values = rolling_mean(vh.values, N)
    return vh


def get_node_depth_v(expt,
                     z1=-200,
                     z2=-800,
                     N=1,
                     mean_axes=(0, 3),
                     **initializer):
    wasinma2 = False
    if 2 in mean_axes:
        wasinma2 = True
        mean_axes = list(mean_axes)
        mean_axes.remove(2)
    with pym6.Dataset(expt.fil1, **initializer) as ds, pym6.Dataset(
            expt.fil2, **initializer) as ds2:
        e = ds2.e.final_loc('vi').yep().read().nanmean(
            axis=mean_axes).move_to('v').compute()
        if expt.z1 is not None:
            z1 = min(z1, expt.z1)
        if expt.z2 is not None:
            z2 = min(z2, expt.z2)
        z = np.linspace(z2, z1, 100)
        v = ds2.v.read().nanmean(axis=mean_axes).toz(z, e).to_DataArray()
        argmax = np.argmin(np.fabs(v.values), axis=1)
        core_depth = xr.DataArray(
            rolling_mean(z[argmax.squeeze()], N),
            dims=('yq', ),
            coords=dict(yq=v.coords['yq']))
    if wasinma2:
        core_depth = core_depth.mean('yq')
    return core_depth


def MoverturnSF(expt):
    with pym6.Dataset(expt.fil2) as ds:
        moc = ds.vh.read().nanmean(axis=0).reduce_(np.sum, axis=3).compute()
        moc.array = np.cumsum(moc.values, axis=1)
        moc = moc.tob(axis=1).to_DataArray().squeeze()
    return moc / 1e6


def ZoverturnSF(expt):
    with pym6.Dataset(expt.fil2) as ds:
        zoc = ds.uh.read().nanmean(axis=0).reduce_(np.sum, axis=2).compute()
        zoc.array = np.cumsum(zoc.values, axis=1)
        zoc = zoc.tob(axis=1).to_DataArray().squeeze()
    return zoc / 1e6


def overturnSwash(expt, axis=3):
    with pym6.Dataset(expt.fil3) as ds:
        deeplay = ds.islayerdeep.isel(
            x=slice(10, 11), z=slice(0, 1), y=slice(0, 1)).read().compute()
        swash = ds.islayerdeep.read().compute()
        swash = ((-swash + deeplay.values) * 100 / deeplay.values).compute()
        swash = swash.nanmean(axis=axis).tob(axis=1).to_DataArray().squeeze()
    return swash


def plot_merid_transport(exps, **initializer):
    """Plots meridional transport at the EB as a function of y"""
    figsize = initializer.get('figsize', (6, 6))
    if 'figsize' in initializer:
        initializer.pop('figsize')
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize)
    fig = exps.plot1d(
        get_ntrans,
        np.greater_equal,
        ax=ax[0],
        N=50,
        sum_axes=(1, 3),
        **initializer)
    fig = exps.plot1d(
        get_ntrans, np.less, ax=ax[1], N=100, sum_axes=(1, 3), **initializer)
    for axc in ax:
        axc.set_title('')
        axc.grid()
    ax[0].set_xlabel('')
    ax[1].legend(loc='best')
    ax[1].set_xlabel('Latitude')
    return fig


def plot_merid_transport_1d(exps, **initializer2):
    """Plots meridional transport at the EB averaged between 34N and 38N"""
    fig = exps.plotpoint(
        get_ntrans,
        np.greater_equal,
        sum_axes=(1, 3),
        south_lat=34,
        north_lat=38,
        method=np.mean,
        plot_kwargs=dict(color='r', marker='*', label='North'),
        **initializer2)
    fig = exps.plotpoint(
        get_ntrans,
        np.less,
        ax=fig.axes[0],
        sum_axes=(1, 3),
        south_lat=34,
        north_lat=38,
        method=np.mean,
        plot_kwargs=dict(color='b', marker='o', label='South'),
        **initializer2)
    fig.axes[0].set_ylabel('Sv')
    fig.axes[0].grid()
    return fig


def plot_current_zc_depth(exps, **initializer):
    """Plots meridional variation of depth of zero-crossing of meridional velocity"""
    figsize = initializer.get('figsize', (6, 3))
    if 'figsize' in initializer:
        initializer.pop('figsize')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig = exps.plot1d(get_node_depth_v, N=100, ax=ax, **initializer)
    ax.legend(loc='best')
    ax.set_xlim(25, 45)
    ax.set_ylabel('Current ZC depth (m)')
    ax.set_xlabel('Latitude')
    return fig


def plot_current_zc_depth_1d(exps, **initializer2):
    """Plots depth of zero-crossing of meridional velocity averaged between 34 and 38N"""
    figsize = initializer2.get('figsize', (6, 3))
    if 'figsize' in initializer2:
        initializer2.pop('figsize')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig = exps.plotpoint(
        get_node_depth_v,
        mean_axes=(0, 2, 3),
        ax=ax,
        south_lat=34,
        north_lat=38,
        plot_kwargs=dict(marker='*'),
        **initializer2)
    ax.set_ylabel('Current ZC depth (m)')
    return fig


def plot_merid_overtsf(exps, figsize=(8, 3)):
    """Plots meridional overturning SF along isopycnals"""
    fig, ax = plt.subplots(
        1, len(exps.list_), sharex=True, sharey=True, figsize=figsize)
    fig = exps.plot2d(
        MoverturnSF,
        plot_kwargs=dict(
            cmap='RdBu_r',
            vmin=-30,
            vmax=30,
            yincrease=True,
            add_colorbar=False),
        ctr_kwargs=dict(
            colors='k',
            yincrease=True,
            add_colorbar=False,
            levels=np.array([-0.5, -0.25, -0.1, 0.25, 0.5, 1, 1.5, 2]) * 1e1),
        contours=True,
        fig=fig)
    #fig.tight_layout()
    #cbar = fig.colorbar(fig.axes[0].collections[0], ax=fig.axes)
    fig = exps.plot2d(
        overturnSwash,
        contourf=False,
        contours=True,
        ctr_kwargs=dict(levels=[1], colors='g', yincrease=True),
        axis=3,
        fig=fig)
    for axc in ax:
        axc.set_ylabel('')
        axc.set_xlabel('Latitude')
    ax[0].set_ylabel(r'b (ms$^{-2}$)')
    return fig


def plot_zonal_overtsf(exps, figsize=(10, 4)):
    """Plots zonal overturning SF along isopycnals"""
    fig, ax = plt.subplots(
        1, len(exps.list_), sharex=True, sharey=True, figsize=figsize)
    fig = exps.plot2d(
        ZoverturnSF,
        plot_kwargs=dict(
            cmap='RdBu_r',
            vmin=-30,
            vmax=30,
            add_colorbar=False,
            yincrease=True),
        ctr_kwargs=dict(
            colors='k',
            yincrease=True,
            levels=np.array([-0.5, -0.25, -0.1, 0.25, 0.5, 1, 1.5, 2]) * 1e1),
        contours=True,
        fig=fig)
    #fig.tight_layout()
    #cbar = fig.colorbar(fig.axes[0].collections[0], ax=fig.axes)
    #fig = exps.plot2d(
    #    overturnSwash,
    #    contourf=False,
    #    contours=True,
    #    ctr_kwargs=dict(levels=[1], colors='g', yincrease=True),
    #    axis=2,
    #    fig=fig)
    for axc in ax:
        axc.set_ylabel('')
        axc.set_xlabel('Longitude')
    ax[0].set_ylabel(r'b (ms$^{-2}$)')
    return fig


def transport_scaling(expt):
    psi = MoverturnSF(expt).max() * 1e6
    return psi * expt.fn / expt.dt**3


def plot_transport_scaling(exps):
    """Plots scaling of transport"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    fig = exps.plotpoint(
        transport_scaling, ax=ax, plot_kwargs=dict(marker='*'))
    ax.set_ylabel(r'$\psi/\Delta T ^3$')
    return fig
