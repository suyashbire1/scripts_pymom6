import pymom6.pymom6 as pym6
import numpy as np
import xarray as xr
import pandas as pd
import importlib
import twamomx_budget as tx
import twamomy_budget as ty
import twapv_budget as tp
importlib.reload(pym6)


def get_thickness_budget(fil1, fil2, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil2, **initializer) as ds2:
        uhx = ds2.uh.xsm().read().nanmean(axis=0).dbyd(
            3, weights='area').compute()
        uhx.name = 'Grad zonal thickness flux'
        uhx.math = r'$(\overline{\sigma u})_{\tilde{x}}$'
        vhy = ds2.vh.ysm().read().nanmean(axis=0).dbyd(
            2, weights='area').compute()
        vhy.name = 'Grad merid thickness flux'
        vhy.math = r'$(\overline{\sigma v})_{\tilde{y}}$'
        dwd = ds2.wd.zep().read().nanmean(axis=0).np_ops(
            np.diff, axis=1, sets_vloc='l').compute()
        dwd.name = 'Grad vert thickness flux'
        dwd.math = r'$(\overline{\sigma \varpi})_{\tilde{b}}$'
    return [uhx, vhy, dwd]


def get_PV_budget(fil1, fil2, **initializer):
    term1 = geostrophic_adv(fil1, fil2, **initializer)
    term2 = frictional(fil1, fil2, **initializer)
    term3 = reynolds_stress(fil1, fil2, **initializer)
    term4 = form_drag(fil1, fil2, **initializer)
    term5 = beta_term(fil1, fil2, **initializer)
    term6 = diapycnal(fil1, fil2, **initializer)
    return [term1, term2, term3, term4, term5, term6]


def extract_budget(fil1, fil2, fil3=None, name='Thickness', **initializer):
    meanax = initializer.get('meanax', 2)
    initializer.pop('meanax')
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    if name == 'PV':
        initializer['final_loc'] = 'qi'
        with pym6.Dataset(fil2, **initializer) as ds:
            e = ds.e.zep().yep().xep().read().move_to('v').move_to(
                'q').nanmean(axis=0).nanmean(axis=meanax).compute()
        initializer['final_loc'] = 'ql'
    elif name == 'Thickness':
        initializer['final_loc'] = 'hi'
        with pym6.Dataset(fil2, **initializer) as ds:
            e = ds.e.zep().read().nanmean(axis=(0, 2)).compute()
        initializer['final_loc'] = 'hl'
    else:
        raise ValueError
    if fil3 is not None:
        with pym6.Dataset(fil3) as ds:
            islaydeepmax = ds.islayerdeep.read().compute(check_loc=False).array
            islaydeepmax = islaydeepmax[0, 0, 0, 0]
        with pym6.Dataset(
                fil3, fillvalue=np.nan, **initializer) as ds, pym6.Dataset(
                    fil1, **initializer) as ds2:
            swash = ds.islayerdeep
            uv = ds2.uv
            swash.indices = uv.indices
            swash.dim_arrays = uv.dim_arrays
            if name == 'PV':
                swash = swash.read()
            elif name == 'Thickness':
                swash = swash.ysm().xsm().read().move_to('v').move_to('h')
            swash.nanmean(axis=0).nanmean(axis=meanax).compute()
            swash = ((-swash + islaydeepmax) / islaydeepmax * 100).compute()
            if z is not None:
                swash = swash.toz(z, e)
            swash = swash.tokm(3).to_DataArray()
    else:
        swash = None
    if name == 'Thickness':
        blist = get_thickness_budget(fil1, fil2, **initializer)
    elif name == 'PV':
        blist = get_PV_budget(fil1, fil2, **initializer)
    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.nanmean(axis=meanax)
        if z is not None:
            b = b.toz(z, e)
        blist[i] = b.tokm(3).to_DataArray(check_loc=False)
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = name + ' budget'
    e = e.tokm(3).to_DataArray()
    return_dict = dict(
        blist_concat=blist_concat, blist=blist, e=e, swash=swash)
    return return_dict


def geostrophic_adv(fil1, fil2, **initializer):
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        eq = ds.e.final_loc('qi').xep().yep().read().move_to('v').move_to(
            'q').nanmean(axis=(0, 2)).compute()
        pfv = ds.PFv.final_loc('ql').xep().read().nanmean(
            axis=0).move_to('q').compute()
        hx = ds1.h_Cv.final_loc('ql').xep().read().nanmean(
            axis=0).dbyd(3).compute()
        f_slice = hx.get_slice_2D()._slice_2D
        f = initializer['geometry'].f[f_slice]
        term1 = ((pfv * hx) / f).compute()
        term1.name = r'$-\bar{m}_{\tilde{y}}\bar{\sigma}_{\tilde{x}}$'

        pfu = ds.PFu.final_loc('ql').yep().read().nanmean(
            axis=0).move_to('q').compute()
        hy = ds1.h_Cu.fillvalue(np.nan).final_loc('ql').yep().read().nanmean(
            axis=0).dbyd(2).compute()
        term2 = (((-pfu).compute() * hy) / f).compute()
        term2.name = r'$\bar{m}_{\tilde{x}}\bar{\sigma}_{\tilde{y}}$'
    term = (term1 + term2).compute()
    term.math = r"$\frac{J(\bar{m},\bar{\sigma})}{f}$"
    term.name = r"Geostrophic advection"
    return term


def frictional(fil1, fil2, **initializer):
    bc_type = dict(
        v=['mirror', 'circsymh', 'zeros', 'circsymq', 'circsymh', 'neumann'])
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdiffv = ds1.twa_hdiffv.bc_type(bc_type).final_loc(
            'ql').xep().read().dbyd(3).nanmean(axis=0).compute()
        hdvdtvisc = ds1.twa_hdvdtvisc.bc_type(bc_type).final_loc(
            'ql').xep().read().dbyd(3).nanmean(axis=0).compute()
        f_slice = hdiffv.get_slice_2D()._slice_2D
        f = initializer['geometry'].f[f_slice]
        term3 = ((hdiffv + hdvdtvisc) / f).compute()
        term3.math = r'$\frac{(\bar{\sigma}\hat{Y})_{\tilde{x}}}{f}$'
        term3.name = r"Frictional creation of PV"
    return term3


def reynolds_stress(fil1, fil2, **initializer):
    initializer = tp.get_domain(fil1, fil2, **initializer)
    bc_type = dict(
        v=['mirror', 'circsymh', 'zeros', 'circsymq', 'circsymh', 'neumann'])
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        h = ds1.h_Cv.bc_type(bc_type).final_loc('ql').xep().read().nanmean(
            axis=0).compute(check_loc=False)
        xdivep = ty.get_xdivep(fil1, fil2, **initializer).xep().bc_type(
            bc_type).implement_BC_if_necessary().multiply_by('dyCv').compute()
        ydivep = ty.get_ydivep(fil1, fil2, **initializer).xep().bc_type(
            bc_type).implement_BC_if_necessary().multiply_by('dyCv').compute()
        term4 = ((xdivep + ydivep) * h).dbyd(
            3, weights='area').compute(check_loc=False)
        f_slice = term4.get_slice_2D()._slice_2D
        f = initializer['geometry'].f[f_slice]
        term4 = (term4 / f).compute(check_loc=False)
        term4.math = r'$\frac{-(\bar{\sigma}\widehat{u^{\prime\prime}v^{\prime\prime}})_{\tilde{x}\tilde{x}}}{f}$'
        term4.name = r"Reynolds stress"
    return term4


def form_drag(fil1, fil2, **initializer):
    bc_type = dict(
        v=['mirror', 'circsymh', 'zeros', 'circsymq', 'circsymh', 'neumann'])
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        h = ds1.h_Cv.bc_type(bc_type).final_loc('ql').xep().read().nanmean(
            axis=0).compute(check_loc=False)
        bdivep = ty.get_bdivep(fil1, fil2, **initializer).xep().bc_type(
            bc_type).implement_BC_if_necessary().multiply_by('dyCv').compute()
        term5 = (bdivep * h).dbyd(3, weights='area').compute(check_loc=False)
        f_slice = term5.get_slice_2D()._slice_2D
        f = initializer['geometry'].f[f_slice]
        term5 = (term5 / f).compute(check_loc=False)
        term5.math = r'$-\frac{(\overline{\zeta^{\prime}m_{\tilde{y}}^{\prime}})_{\tilde{x}\tilde{b}}}{f}$'
        term5.name = r"Form drag"
    return term5


def beta(y):
    omega = 2 * np.pi / 24 / 3600 + 2 * np.pi / 365 / 24 / 3600
    R = 6378e3
    return 2 * omega * np.cos(np.radians(y)) / R


def beta_term(fil1, fil2, **initializer):
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        vh = ds.vh.final_loc('ql').xep().read().divide_by('dxCv').nanmean(
            axis=0).move_to('q').compute()
        y = (initializer['south_lat'] + initializer['north_lat']) / 2
        f_slice = vh.get_slice_2D()._slice_2D
        f = initializer['geometry'].f[f_slice]
        bsv = ((-vh).compute() * beta(y) / f).compute()
        bsv.math = r'$ -\frac{\beta\bar{\sigma}\hat{v}}{f}$'
        bsv.name = r"beta term"
    return bsv


def diapycnal(fil1, fil2, **initializer):
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        dwd = ds.wd.final_loc('ql').xep().yep().zep().read().nanmean(
            axis=0).np_ops(
                np.diff, axis=1,
                sets_vloc='l').move_to('u').move_to('q').compute()
        term7 = (-dwd).compute()
        term7.math = r'$(\bar{\sigma}\hat{\varpi})_{\tilde{b}}$'
        term7.name = r"Diapycnal term"
    return term7
