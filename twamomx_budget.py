import pymom6.pymom6 as pym6
import numpy as np
import importlib
import xarray as xr
import pandas as pd
import string
importlib.reload(pym6)


def get_h(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1:
        h = ds1.h_Cu.read().nanmean(axis=0).compute()
        a = h.values
        a[a < htol] = np.nan
        h.values = a
        return h


def get_ur(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        ur = ds2.uh.read().nanmean(axis=0).divide_by('dyCu') / h
        ur = ur.compute()
        return ur


def get_advx(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    ur = get_ur(fil1, fil2, **initializer)
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_forx = ds1.h_Cu.xsm().xep().read().nanmean(axis=0).compute()
        a = h_forx.values
        a[a < htol] = np.nan
        h_forx.values = a
        urx = ds2.uh.xsm().xep().read().nanmean(
            axis=0).divide_by('dyCu') / h_forx
        urx = urx.dbyd(3).move_to('u').compute()
        advx = (-urx * ur).compute()
        advx.name = 'Zonal advection'
        advx.math = r'$-\hat{u}\hat{u}_{\tilde{x}}$'
        advx.units = r'ms$^{-2}$'
        return advx


def get_advy(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h = ds1.h_Cu.read().nanmean(axis=0).compute()
        h_fory = ds1.h_Cu.ysm().yep().read().nanmean(axis=0).compute()
        a = h_fory.values
        a[a < htol] = np.nan
        h_fory.values = a
        ury = ds2.uh.ysm().yep().read().nanmean(
            axis=0).divide_by('dyCu') / h_fory
        ury = ury.dbyd(2).move_to('u').compute()
        hvm = ds2.vh.ysm().xep().read().nanmean(
            axis=0).divide_by('dxCv').move_to('h').move_to('u').compute()
        advy = (-hvm * ury / h).compute()
        advy.name = 'Meridional advection'
        advy.math = r'$-\hat{v}\hat{u}_{\tilde{y}}$'
        advy.units = r'ms$^{-2}$'
        return advy


def get_advb(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    htol = initializer.get('htol', 1e-3)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_forb = ds1.h_Cu.zsm().zep().read().nanmean(axis=0).compute()
        a = h_forb.values
        a[a < htol] = np.nan
        h_forb.values = a
        urb = ds2.uh.zsm().zep().read().nanmean(axis=0) / h_forb
        urb = urb.divide_by('dyCu').dbyd(1).move_to('l').compute()
        db = np.diff(ds2.zl)[0] * 9.8 / 1031
        hwm = ds2.wd.xep().zep().read().nanmean(
            axis=0).move_to('l').move_to('u') * db
        advb = (-hwm * urb / h).compute()
        advb.name = 'Vertical advection'
        advb.math = r'$-\hat{\varpi}\hat{u}_{\tilde{b}}$'
        advb.units = r'ms$^{-2}$'
        return advb


def get_cor(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hfvm = ds1.twa_hfv.read().nanmean(axis=0).compute()
        cor = (hfvm / h).compute()
        cor.name = 'Coriolis force'
        cor.math = r'$f\hat{v}$'
        cor.units = r'ms$^{-2}$'
        return cor


def get_pfum(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        pfum = ds2.PFu.read().nanmean(axis=0).compute()
        pfum = h * pfum / h
        pfum = pfum.compute()
        pfum.name = 'Grad of Montg Pot'
        pfum.math = r'$-\bar{m}_{\tilde{x}}$'
        pfum.units = r'ms$^{-2}$'
        return pfum


def get_xdivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        huuxm = ds1.huu_Cu.xsm().xep().read().divide_by('dyCu').dbyd(
            3).nanmean(axis=0).move_to('u').compute()
        xdivep1 = (-huuxm / h).compute()
        return xdivep1


def get_xdivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    ur = get_ur(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        humx = ds2.uh.xsm().xep().read().nanmean(
            axis=0).divide_by('dyCu').dbyd(3).move_to('u').compute()
        xdivep3 = (humx * ur / h).compute()
        return xdivep3


def get_edlsqmx(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        esq = ds1.esq.xep().read().nanmean(axis=0)
        e = ds2.e.xep().zep().read().nanmean(axis=0).move_to('l')**2
        e = e.compute(check_loc=False)
        edlsqmx = esq - e
        edlsqmx = edlsqmx.dbyd(3).compute()
        return edlsqmx


def get_xdivep4(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    edlsqmx = get_edlsqmx(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        db = -np.diff(ds1.zl)[0] * 9.8 / 1031
        xdivep4 = (-0.5 * edlsqmx * db / h).compute()
        return xdivep4


def get_xdivep(fil1, fil2, **initializer):
    xdivep1 = get_xdivep1(fil1, fil2, **initializer)
    xdivep2 = -get_advx(fil1, fil2, **initializer)
    xdivep3 = get_xdivep3(fil1, fil2, **initializer)
    xdivep4 = get_xdivep4(fil1, fil2, **initializer)
    xdivep = ((xdivep1 + xdivep2).compute() +
              (xdivep3 + xdivep4).compute()).compute()
    xdivep.name = 'Div of zonal EP flux'
    xdivep.math = (
        r"""-$\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
        r"""\widehat{u ^{\prime \prime} u ^{\prime \prime} })_{\tilde{x}}$"""
        r"""-$\frac{1}{2\bar{\sigma}}"""
        r"""(\bar{\zeta ^{\prime 2}})_{\tilde{x}}$""")
    xdivep.units = r'ms$^{-2}$'
    return xdivep


def get_ydivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        huuxpt = ds1.twa_huuxpt.read().nanmean(axis=0).compute()
        huvymt = ds1.twa_huvymt.read().nanmean(axis=0).compute()
        huuxphuvym = huuxpt + huvymt
        huuxphuvym = huuxphuvym.compute()
        huuxm = ds1.huu_Cu.xsm().xep().read().divide_by('dyCu').dbyd(
            3).nanmean(axis=0).move_to('u').compute()
        huvym = huuxphuvym + huuxm
        ydivep1 = (huvym / h).compute()
        return ydivep1


def get_ydivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    ur = get_ur(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        hvmy = ds2.vh.ysm().xep().read().nanmean(
            axis=0).divide_by('dxCv').dbyd(2).move_to('u').compute()
        ydivep3 = (hvmy * ur / h).compute()
        return ydivep3


def get_ydivep(fil1, fil2, **initializer):
    ydivep1 = get_ydivep1(fil1, fil2, **initializer)
    ydivep2 = -get_advy(fil1, fil2, **initializer)
    ydivep3 = get_ydivep3(fil1, fil2, **initializer)
    ydivep = (ydivep1 + ydivep2 + ydivep3).compute()
    ydivep.name = 'Div of merid EP flux'
    ydivep.math = (r"""-$\frac{1}{\bar{\sigma}}"""
                   r"""(\bar{\sigma}\widehat{u ^{\prime \prime} """
                   r"""v ^{\prime \prime}})_{\tilde{y}}$""")
    ydivep.units = r'ms$^{-2}$'
    return ydivep


def get_bdivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        huwbm = ds1.twa_huwb.read().nanmean(axis=0).compute()
        bdivep1 = (huwbm / h).compute()
        return bdivep1


def get_bdivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    ur = get_ur(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        hwb = ds2.wd.xep().zep().read().nanmean(axis=0).move_to('u').np_ops(
            np.diff, axis=1, sets_vloc='l').compute()
        bdivep3 = (hwb * ur / h).compute()
        return bdivep3


def get_bdivep4(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    edlsqmx = get_edlsqmx(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        hpfu = ds1.twa_hpfu.read().nanmean(axis=0).compute()
        pfum = ds2.PFu.read().nanmean(axis=0).compute()
        db = -np.diff(ds1.zl)[0] * 9.8 / 1031
        edpfudmb = -(-hpfu + (h * pfum).compute() -
                     (edlsqmx * db * 0.5).compute()) / h
        bdivep4 = edpfudmb.compute()
        return bdivep4


def get_bdivep(fil1, fil2, **initializer):
    bdivep1 = get_bdivep1(fil1, fil2, **initializer)
    bdivep2 = -get_advb(fil1, fil2, **initializer)
    bdivep3 = get_bdivep3(fil1, fil2, **initializer)
    bdivep4 = get_bdivep4(fil1, fil2, **initializer)
    bdivep = (bdivep1 + bdivep2 + bdivep3 + bdivep4).compute()
    bdivep.name = 'Form Drag'
    bdivep.math = (r"""-$\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
                   r"""\widehat{u ^{\prime \prime} """
                   r"""\varpi ^{\prime \prime}})_{\tilde{b}}$"""
                   r""" -$\frac{1}{2\bar{\sigma}}"""
                   r"""(\overline{\zeta ^\prime m_{\tilde{x}}^\prime})"""
                   r"""_{\tilde{b}}$""")
    bdivep.units = r'ms$^{-2}$'
    return bdivep


def get_X1twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdiffum = ds1.twa_hdiffu.read().nanmean(axis=0).compute()
        X1twa = (hdiffum / h).compute()
        X1twa.name = 'Horizontal diffusion'
        X1twa.math = r'$\widehat{X^H}$'
        X1twa.units = r'ms$^{-2}$'
        return X1twa


def get_X2twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdudtviscm = ds1.twa_hdudtvisc.read().nanmean(axis=0).compute()
        X2twa = (hdudtviscm / h).compute()
        X2twa.name = 'Vertical viscous forces'
        X2twa.math = r'$\widehat{X^V}$'
        X2twa.units = r'ms$^{-2}$'
        return X2twa


def get_BxplusPVflux(fil1, fil2, **initializer):
    advx = get_advx(fil1, fil2, **initializer)
    advy = get_advy(fil1, fil2, **initializer)
    cor = get_cor(fil1, fil2, **initializer)
    pfum = get_pfum(fil1, fil2, **initializer)
    BxplusPVflux = (advx + advy + cor + pfum).compute()
    return BxplusPVflux


def get_PV(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    htol = initializer.get('htol', 1e-3)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_fory = ds1.h_Cu.ysm().yep().read().nanmean(axis=0).compute()
        a = h_fory.values
        a[a < htol] = np.nan
        h_fory.values = a
        ury = ds2.uh.ysm().yep().read().nanmean(
            axis=0).divide_by('dyCu') / h_fory
        ury = ury.dbyd(2)
        f_slice = ury.get_slice_2D()._slice_2D
        ury = ury.move_to('u').compute()
        h_forx = ds1.h_Cv.xep().ysm().read().nanmean(axis=0).compute(
            check_loc=False)
        a = h_forx.values
        a[a < htol] = np.nan
        h_forx.values = a
        vrx = ds2.vh.xep().ysm().read().nanmean(
            axis=0).divide_by('dxCv') / h_forx
        vrx = vrx.dbyd(3).move_to('u').compute()
        f = initializer['geometry'].f[f_slice]
        f = 0.5 * (f[:-1] + f[1:])
        PV = ((vrx - ury + f) / h).compute()
        PV.name = r'PV$^\sharp$'
        return PV


def get_PVflux(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    PV = get_PV(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        vh = ds2.vh.xep().ysm().read().divide_by('dxCv').nanmean(
            axis=0).move_to('h').move_to('u').compute()
        PVflux = (PV * vh).compute()
        PVflux.name = 'PV flux'
        PVflux.math = r'$\bar{\sigma}\hat{v}\Pi^{\sharp}$'
        PVflux.units = r'ms$^{-2}$'
        return PVflux


def get_Bx(fil1, fil2, **initializer):
    PVflux = get_PVflux(fil1, fil2, **initializer)
    BxplusPVflux = get_BxplusPVflux(fil1, fil2, **initializer)
    Bx = (BxplusPVflux - PVflux).compute()
    Bx.name = 'Grad Bernoulli func'
    Bx.math = r'$-\bar{B}_{\tilde{x}}$'
    Bx.units = r'ms$^{-2}$'
    return Bx


def extract_twamomx_terms(fil1, fil2, **initializer):

    conventional_list = [
        get_advx, get_advy, get_advb, get_cor, get_pfum, get_xdivep,
        get_ydivep, get_bdivep, get_X1twa, get_X2twa
    ]

    withPVflux_list = [
        get_PVflux, get_advb, get_Bx, get_xdivep, get_ydivep, get_bdivep,
        get_X1twa, get_X2twa
    ]

    only = initializer.get('only', range(len(conventional_list)))
    type = initializer.get('type', 'conventional')
    if 'type' in initializer:
        initializer.pop('type')
    if 'only' in initializer:
        initializer.pop('only')
    return_list = []
    if type == 'conventional':
        for i, func in enumerate(conventional_list):
            if i in only:
                return_list.append(func(fil1, fil2, **initializer))
    elif type == 'withPVflux':
        for i, func in enumerate(withPVflux_list):
            if i in only:
                return_list.append(func(fil1, fil2, **initializer))
    return return_list


def extract_budget(fil1, fil2, **initializer):
    meanax = initializer.get('meanax', 2)
    initializer.pop('meanax')
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'ui'
    with pym6.Dataset(fil2, **initializer) as ds:
        e = ds.e.xep().zep().read().nanmean(axis=(0, 2)).move_to('u').compute()
    initializer['final_loc'] = 'ul'
    blist = extract_twamomx_terms(fil1, fil2, **initializer)
    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.nanmean(axis=meanax)
        if z is not None:
            b = b.toz(z, e)
        blist[i] = b.to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Zonal Momentum Budget'
    return blist_concat, blist


def plot_budget(fil1, fil2, **initializer):
    blist_concat, blist = extract_budget(fil1, fil2, **initializer)
    fg = blist_concat.plot.imshow(
        'xq',
        'z',
        size=2,
        aspect=(1 + np.sqrt(5)) / 2,
        yincrease=True,
        vmin=-7e-5,
        vmax=7e-5,
        cmap='RdBu_r',
        col='Term',
        col_wrap=2)
    for i, ax in enumerate(fg.axes.flat):
        ax.set_title('(' + string.ascii_lowercase[i] + ') ' + blist[i].name)
        ax.text(-0.45, -1000, blist[i].attrs['math'], fontsize=15)
    fg.cbar.formatter.set_powerlimits((0, 0))
    fg.cbar.update_ticks()
    return fg


if __name__ == '__main__':
    import sys
    fil = sys.argv[1]
    initializer = dict(sys.argv[2])
    extract_twamomx_terms(fil, initializer)