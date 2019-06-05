import numpy as np
import xarray as xr
import pymom6.pymom6 as pym6
import matplotlib.pyplot as plt
import moc as ty
import zoc as tx
import zoc_from_vmom as zb
import moc_from_umom as mb
import pandas as pd
import importlib
importlib.reload(ty)
importlib.reload(tx)
importlib.reload(zb)
importlib.reload(mb)
importlib.reload(pym6)


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


def baroSF(expt):
    with pym6.Dataset(expt.fil2) as ds:
        psiB = ds.uh.read().nanmean(axis=0).reduce_(np.sum, axis=1).compute()
        psiB.array = -np.cumsum(psiB.values, axis=2)
        psiB = psiB.to_DataArray().squeeze()
    return psiB / 1e6


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


def MoverturnSFmean(expt):
    with pym6.Dataset(
            expt.fil2,
            geometry=expt.geometry) as ds, pym6.Dataset(expt.fil1) as ds1:
        moc = ds.v.read().nanmean(axis=0).multiply_by('dxCv')
        h = ds1.h_Cv.read().nanmean(axis=0).compute()
        moc = (moc * h).reduce_(np.sum, axis=3).compute()
        moc.array = np.cumsum(moc.values, axis=1)
        moc = moc.tob(axis=1).to_DataArray().squeeze()
    return moc / 1e6


def ZoverturnSFmean(expt):
    with pym6.Dataset(
            expt.fil2,
            geometry=expt.geometry) as ds, pym6.Dataset(expt.fil1) as ds1:
        zoc = ds.u.read().nanmean(axis=0).multiply_by('dyCu')
        h = ds1.h_Cu.read().nanmean(axis=0).compute()
        zoc = (zoc * h).reduce_(np.sum, axis=2).compute()
        zoc.array = np.cumsum(zoc.values, axis=1)
        zoc = zoc.tob(axis=1).to_DataArray().squeeze()
    return zoc / 1e6


def MoverturnSFeddy(expt):
    mocres = MoverturnSF(expt)
    mocmean = MoverturnSFmean(expt)
    return mocres - mocmean


def ZoverturnSFeddy(expt):
    zocres = ZoverturnSF(expt)
    zocmean = ZoverturnSFmean(expt)
    return zocres - zocmean


def MoverturnSFstanding(expt):
    with pym6.Dataset(
            expt.fil2,
            geometry=expt.geometry) as ds, pym6.Dataset(expt.fil1) as ds1:
        v = ds.v.read().nanmean(axis=0).compute()
        v.values = v.values - np.mean(v.values, axis=3, keepdims=True)
        v = v.multiply_by('dxCv')
        h = ds1.h_Cv.read().nanmean(axis=0).compute()
        h.values = h.values - np.mean(h.values, axis=3, keepdims=True)
        moc = (v * h).reduce_(np.sum, axis=3).compute()
        moc.array = np.cumsum(moc.values, axis=1)
        moc = moc.tob(axis=1).to_DataArray().squeeze()
    return moc / 1e6


def MoverturnSFtmeanzmean(expt):
    with pym6.Dataset(
            expt.fil2,
            geometry=expt.geometry) as ds, pym6.Dataset(expt.fil1) as ds1:
        v = ds.v.read().nanmean(axis=0).compute()
        v.values = np.repeat(
            np.mean(v.values, axis=3, keepdims=True), v.shape[3], axis=3)
        v = v.multiply_by('dxCv')
        h = ds1.h_Cv.read().nanmean(axis=0).compute()
        h.values = np.repeat(
            np.mean(h.values, axis=3, keepdims=True), h.shape[3], axis=3)
        moc = (v * h).reduce_(np.sum, axis=3).compute()
        moc.array = np.cumsum(moc.values, axis=1)
        moc = moc.tob(axis=1).to_DataArray().squeeze()
    return moc / 1e6


def ZoverturnSFstanding(expt):
    with pym6.Dataset(
            expt.fil2,
            geometry=expt.geometry) as ds, pym6.Dataset(expt.fil1) as ds1:
        u = ds.u.read().nanmean(axis=0).compute()
        u.values = u.values - np.mean(u.values, axis=2, keepdims=True)
        u = u.multiply_by('dyCu')
        h = ds1.h_Cu.read().nanmean(axis=0).compute()
        h.values = h.values - np.mean(h.values, axis=2, keepdims=True)
        zoc = (u * h).reduce_(np.sum, axis=2).compute()
        zoc.array = np.cumsum(zoc.values, axis=1)
        zoc = zoc.tob(axis=1).to_DataArray().squeeze()
    return zoc / 1e6


def ZoverturnSFtmeanzmean(expt):
    with pym6.Dataset(
            expt.fil2,
            geometry=expt.geometry) as ds, pym6.Dataset(expt.fil1) as ds1:
        u = ds.u.read().nanmean(axis=0).compute()
        u.values = np.repeat(
            np.mean(u.values, axis=2, keepdims=True), u.shape[2], axis=2)
        u = u.multiply_by('dyCu')
        h = ds1.h_Cu.read().nanmean(axis=0).compute()
        h.values = np.repeat(
            np.mean(h.values, axis=2, keepdims=True), h.shape[2], axis=2)
        zoc = (u * h).reduce_(np.sum, axis=2).compute()
        zoc.array = np.cumsum(zoc.values, axis=1)
        zoc = zoc.tob(axis=1).to_DataArray().squeeze()
    return zoc / 1e6


def ZonalSectionofZOCbudget(expt, **initializer):
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'ui'
    with pym6.Dataset(expt.fil2, **initializer) as ds:
        e = ds.e.xep().zep().read().move_to('u').nanmean(axis=(0, 2)).compute()
    initializer['final_loc'] = 'ul'
    blist = tx.extract_twamomx_terms(expt.fil1, expt.fil2, **initializer)

    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        #b = b.reduce_(np.sum, axis=2)
        if z is not None:
            b = b.toz(z, e)
        else:
            b.values = np.cumsum(b.values, axis=1)
            b = b.tob(axis=1)
            if i == 0:
                e = e.tob(axis=1)
        blist[i] = b.to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Zonal momentum budget'
    e = e.to_DataArray()
    return_dict = dict(
        blist_concat=blist_concat, blist=blist, e=None, swash=None)
    return return_dict


def ZonalSectionofZOCcomponents(expt, **initializer):
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'vi'
    with pym6.Dataset(expt.fil2, **initializer) as ds:
        e = ds.e.yep().zep().read().move_to('v').nanmean(axis=(0, 2)).compute()
    initializer['final_loc'] = 'vl'
    blist = zb.extract_twamomy_terms(expt.fil1, expt.fil2, **initializer)

    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        #b = b.reduce_(np.sum, axis=2)
        if z is not None:
            b = b.toz(z, e)
        else:
            b.values = np.cumsum(b.values, axis=1)
            b = b.tob(axis=1)
            if i == 0:
                e = e.tob(axis=1)
        blist[i] = b.to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Zonal OC budget'
    e = e.to_DataArray()
    return_dict = dict(blist_concat=blist_concat, blist=blist, e=e, swash=None)
    return return_dict


def ZonalSectionofZOCbudgetfdrel(expt, **initializer):
    blist = zb.extract_twamomy_terms(
        expt.fil1, expt.fil2, only=[3, 6], **initializer)
    for b in blist:
        b.values = np.cumsum(b.values, axis=1)
    fd_zoc = (blist[1] / blist[0]).compute()
    fd_zoc = fd_zoc.tob(axis=1).to_DataArray()
    return fd_zoc.squeeze()


def MeridSectionofMOCbudget(expt, **initializer):
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'vi'
    with pym6.Dataset(expt.fil2, **initializer) as ds:
        e = ds.e.yep().zep().read().move_to('v').nanmean(axis=(0, 3)).compute()
    initializer['final_loc'] = 'vl'
    blist = ty.extract_twamomy_terms(expt.fil1, expt.fil2, **initializer)

    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        # b = b.reduce_(np.sum, axis=3)
        if z is not None:
            b = b.toz(z, e)
        else:
            b.values = np.cumsum(b.values, axis=1)
            b = b.tob(axis=1)
            if i == 0:
                e = e.tob(axis=1)
        blist[i] = b.to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Merid momentum budget'
    e = e.to_DataArray()
    return_dict = dict(
        blist_concat=blist_concat, blist=blist, e=None, swash=None)
    return return_dict


def MeridSectionofMOCcomponents(expt, **initializer):
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'ui'
    with pym6.Dataset(expt.fil2, **initializer) as ds:
        e = ds.e.xep().zep().read().move_to('u').nanmean(axis=(0, 3)).compute()
    initializer['final_loc'] = 'ul'
    blist = mb.extract_twamomx_terms(expt.fil1, expt.fil2, **initializer)

    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        #b = b.reduce_(np.sum, axis=2)
        if z is not None:
            b = b.toz(z, e)
        else:
            b.values = np.cumsum(b.values, axis=1)
            b = b.tob(axis=1)
        blist[i] = (-b).to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Merid OC budget'
    if z is None:
        e = e.tob(axis=1)
    e = e.to_DataArray()
    return_dict = dict(blist_concat=blist_concat, blist=blist, e=e, swash=None)
    return return_dict


def MeridSectionofMOCbudgetfdrel(expt, **initializer):
    blist = mb.extract_twamomx_terms(
        expt.fil1, expt.fil2, only=[3, 6], **initializer)
    for b in blist:
        b.values = np.cumsum(b.values, axis=1)
    fd_zoc = (blist[1] / blist[0]).compute()
    fd_zoc = fd_zoc.tob(axis=1).to_DataArray()
    return fd_zoc.squeeze()


def ZonalSectionofZonalconvMomBudget(expt, **initializer):
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'ui'
    with pym6.Dataset(expt.fil2, **initializer) as ds:
        e = ds.e.xep().zep().read().move_to('u').nanmean(axis=(0, 2)).compute()
    initializer['final_loc'] = 'ul'
    blist = tx.extract_momx_terms(expt.fil1, expt.fil2, **initializer)

    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.reduce_(np.sum, axis=2)
        if z is not None:
            b = b.toz(z, e)
        blist[i] = b.to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Zonal momentum budget'
    e = e.to_DataArray()
    return_dict = dict(blist_concat=blist_concat, blist=blist, e=e, swash=None)
    return return_dict


def MoverturnSFscaled(expt):
    with pym6.Dataset(expt.fil2) as ds:
        moc = ds.vh.read().nanmean(axis=0).reduce_(np.sum, axis=3).compute()
        moc.array = np.cumsum(moc.values, axis=1)
        moc = moc.tob(axis=1).to_DataArray().squeeze()
        db = expt.db
        nsq = expt.nsq
        fn = expt.fn
    return moc / (db**3 / fn / nsq**2)


def ZoverturnSFscaled(expt):
    with pym6.Dataset(expt.fil2) as ds:
        zoc = ds.uh.read().nanmean(axis=0).reduce_(np.sum, axis=2).compute()
        zoc.array = np.cumsum(zoc.values, axis=1)
        zoc = zoc.tob(axis=1).to_DataArray().squeeze()
        db = expt.db
        nsq = expt.nsq
        fn = expt.fn
    return zoc / (db**3 / fn / nsq**2)


def overturnSwash(expt, axis=3):
    with pym6.Dataset(expt.fil3) as ds:
        deeplay = ds.islayerdeep.isel(
            x=slice(10, 11), z=slice(0, 1), y=slice(0, 1)).read().compute()
        swash = ds.islayerdeep.read().compute()
        swash = ((-swash + deeplay.values) * 100 / deeplay.values).compute()
        if axis:
            swash = swash.nanmean(axis=axis)
        swash = swash.tob(axis=1).to_DataArray().squeeze()
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


def plot_merid_overtsfscaled(exps, figsize=(8, 3)):
    """Plots meridional overturning SF along isopycnals"""
    fig, ax = plt.subplots(
        1, len(exps.list_), sharex=True, sharey=True, figsize=figsize)
    fig = exps.plot2d(
        MoverturnSFscaled,
        plot_kwargs=dict(
            cmap='RdBu_r',
            vmin=-0.12,
            vmax=0.12,
            yincrease=True,
            add_colorbar=False),
        ctr_kwargs=dict(
            colors='k',
            yincrease=True,
            add_colorbar=False,
            levels=np.array([-0.5, -0.25, -0.1, 0.25, 0.5, 1, 1.5, 2]) * 1e-1),
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


def plot_zonal_overtsfscaled(exps, figsize=(10, 4)):
    """Plots zonal overturning SF along isopycnals"""
    fig, ax = plt.subplots(
        1, len(exps.list_), sharex=True, sharey=True, figsize=figsize)
    fig = exps.plot2d(
        ZoverturnSFscaled,
        plot_kwargs=dict(
            cmap='RdBu_r',
            vmin=-0.08,
            vmax=0.08,
            add_colorbar=False,
            yincrease=True),
        ctr_kwargs=dict(
            colors='k',
            yincrease=True,
            levels=np.array([-0.5, -0.25, -0.1, 0.25, 0.5, 1, 1.5, 2]) * 1e-1),
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


def get_trans(expt, initializer, fn=np.greater_equal):
    with pym6.Dataset(expt.fil2, final_loc='vl', **initializer) as ds:
        vh = ds.vh.read().nanmean(axis=0).where(
            fn, 0, y=0).reduce_(
                np.sum, axis=1).compute()
    return vh.tokm(3).to_DataArray() / 1e6


def get_core(v, ylims=(25, 55)):
    """Returns the zonal location of the maximum velocity (xmax)."""
    yq = v.coords['yq'].values
    xh = v.coords['x (km)'].values
    cond = (yq >= ylims[0]) & (yq <= ylims[1])
    array = np.fabs(v.values)
    indmax = np.argmax(array, axis=3)
    xmax = xh[indmax[:, :, cond]].squeeze()
    return xr.DataArray(xmax, dims='yq', coords=dict(yq=yq[cond]))


def replace_outlier_with_median(d, data, perc=2):
    dmin, dmax = np.percentile(data, (perc, 100 - perc))
    if d < dmin or d > dmax:
        d = np.median(data)
    return d


def piecewise_replace(data, window=100, perc=2):
    new_data = data.copy()
    for i in range(data.size):
        surround_data = data[max(0, i -
                                 window // 2):min(data.size, i + window // 2)]
        new_data[i] = replace_outlier_with_median(
            data[i], surround_data, perc=perc)
    return new_data


def get_bandwidth(v, ylims=(25, 55), perc=2):
    """Returns the zonal location of where velocity is half of the maximum (xbw)."""
    yq = v.coords['yq'].values
    xh = v.coords['x (km)'].values
    cond = (yq >= ylims[0]) & (yq <= ylims[1])
    array = np.fabs(v.values[:, :, cond, :])
    indmax = np.argmax(array, axis=3).squeeze()
    arraymax = np.amax(array, axis=3)
    arraybw = array - arraymax[:, :, :, np.newaxis] / 2
    xbw = []
    for j in range(yq[cond].size):
        amax = arraymax[:, :, j].squeeze()
        a = arraybw[:, :, j, 0:indmax[j]].squeeze()
        a = a[::-1]
        a1 = a[1:] * a[:-1]
        indzc = np.argmin(a1)
        indzc = a.size - indzc - 1
        xbw.append(xh[indzc])
    #return xr.DataArray(xbw, dims='yq', coords=dict(yq=yq[cond]))
    xbw = np.array(xbw)
    return xr.Dataset(
        data_vars=dict(
            xbw=('yq', xbw.copy()),
            xbw_corrected=('yq', piecewise_replace(xbw, perc=perc))),
        coords=dict(yq=yq[cond]))


def plot_width(expt,
               ax,
               initializer_broad,
               fn=np.greater_equal,
               ylims=(25, 55),
               perc=1):
    vh = get_trans(expt, initializer_broad, fn=fn)
    xmax = get_core(vh, ylims=ylims)
    xbw = get_bandwidth(vh, perc=perc, ylims=ylims)

    im = vh.plot(
        ax=ax, cmap='RdBu_r', add_colorbar=False, vmax=0.85, vmin=-0.85)
    ax.plot(xmax.values, xmax.yq, 'k--', label='_nolegend_')
    ax.plot(xbw.xbw.values, xbw.yq, 'k-', label='_nolegend_')
    ax.plot(xbw.xbw_corrected.values, xbw.yq, 'b-', label='_nolegend_')
    return im


def plot_widths(expt, ax, **initializer):
    im = plot_width(
        expt, ax[0], initializer, ylims=expt.ylimsp, perc=expt.percp)
    im = plot_width(
        expt,
        ax[1],
        initializer,
        fn=np.less,
        ylims=expt.ylimsm,
        perc=expt.percm)
    for axc in ax:
        axc.set_title(expt.name)
    return im


def plot_widths_expts(exps, **initializer):
    fig, ax = plt.subplots(
        1, 2 * len(exps.list_), figsize=(16, 4), sharex=True, sharey=True)
    for i, exp in enumerate(exps.list_):
        im = plot_widths(exp, ax=ax[2 * i:2 * i + 2], **initializer)
    fig.colorbar(im, ax=ax.ravel().tolist())
    for axc in ax:
        axc.set_ylabel('')
    ax[0].set_ylabel('Latitude')
    return fig
