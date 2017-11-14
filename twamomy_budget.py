import pymom6.pymom6 as pym6
import numpy as np
import importlib
import xarray as xr
import pandas as pd
import string
importlib.reload(pym6)


def get_h(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1:
        h = ds1.h_Cv.read().nanmean(axis=0).compute()
        a = h.values
        a[a < htol] = np.nan
        h.values = a
        return h


def get_vr(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        vr = ds2.vh.read().nanmean(axis=0).divide_by('dxCv') / h
        vr = vr.compute()
        return vr


def get_advx(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    htol = initializer.get('htol', 1e-3)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_forx = ds1.h_Cv.xsm().xep().read().nanmean(axis=0).compute()
        a = h_forx.values
        a[a < htol] = np.nan
        h_forx.values = a
        vrx = ds2.vh.xsm().xep().read().nanmean(
            axis=0).divide_by('dxCv') / h_forx
        vrx = vrx.dbyd(3).move_to('v').compute()
        hum = ds2.uh.yep().xsm().read().nanmean(
            axis=0).divide_by('dyCu').move_to('h').move_to('v').compute()
        advx = (-vrx * hum / h).compute()
        advx.name = 'Zonal advection'
        advx.math = r'$-\hat{u}\hat{v}_{\tilde{x}}$'
        advx.units = r'ms$^{-2}$'
        return advx


def get_advy(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    htol = initializer.get('htol', 1e-3)
    vr = get_vr(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_fory = ds1.h_Cv.ysm().yep().read().nanmean(axis=0).compute()
        a = h_fory.values
        a[a < htol] = np.nan
        h_fory.values = a
        vry = ds2.vh.ysm().yep().read().nanmean(
            axis=0).divide_by('dxCv') / h_fory
        vry = vry.dbyd(2).move_to('v').compute()
        advy = (-vr * vry).compute()
        advy.name = 'Meridional advection'
        advy.math = r'$-\hat{v}\hat{v}_{\tilde{y}}$'
        advy.units = r'ms$^{-2}$'
        return advy


def get_advb(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    htol = initializer.get('htol', 1e-3)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_forb = ds1.h_Cv.zsm().zep().read().nanmean(axis=0).compute()
        a = h_forb.values
        a[a < htol] = np.nan
        h_forb.values = a
        vrb = ds2.vh.zsm().zep().read().nanmean(
            axis=0).divide_by('dxCv') / h_forb
        vrb = vrb.dbyd(1).move_to('l').compute()
        db = np.diff(ds2.zl)[0] * 9.8 / 1031
        hwm = ds2.wd.yep().zep().read().nanmean(
            axis=0).move_to('l').move_to('v') * db
        advb = (-hwm * vrb / h).compute()
        advb.name = 'Vertical advection'
        advb.math = r'$-\hat{\varpi}\hat{v}_{\tilde{b}}$'
        advb.units = r'ms$^{-2}$'
        return advb


def get_cor(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hmfum = ds1.twa_hmfu.read().nanmean(axis=0).compute()
        cor = (hmfum / h).compute()
        cor.name = 'Coriolis force'
        cor.math = r'$-f\hat{u}$'
        cor.units = r'ms$^{-2}$'
        return cor


def get_pfvm(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        pfvm = ds2.PFv.read().nanmean(axis=0).compute()
        pfvm = h * pfvm / h
        pfvm = pfvm.compute()
        pfvm.name = 'Grad of Montg Pot'
        pfvm.math = r'$-\bar{m}_{\tilde{y}}$'
        pfvm.units = r'ms$^{-2}$'
        return pfvm


def get_xdivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        huvxpt = ds1.twa_huvxpt.read().nanmean(axis=0).compute()
        hvvymt = ds1.twa_hvvymt.read().nanmean(axis=0).compute()
        huvxphvvym = huvxpt + hvvymt
        huvxphvvym = huvxphvvym.compute()
        hvvym = ds1.hvv_Cv.ysm().yep().read().divide_by('dxCv').dbyd(
            2).nanmean(axis=0).move_to('v').compute()
        huvxm = huvxphvvym + hvvym
        xdivep1 = (huvxm / h).compute()
        return xdivep1


def get_xdivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    vr = get_vr(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        humx = ds2.uh.yep().xsm().read().nanmean(
            axis=0).divide_by('dyCu').dbyd(3).move_to('v').compute()
        xdivep3 = (humx * vr / h).compute()
        return xdivep3


def get_xdivep(fil1, fil2, **initializer):
    xdivep1 = get_xdivep1(fil1, fil2, **initializer)
    xdivep2 = -get_advx(fil1, fil2, **initializer)
    xdivep3 = get_xdivep3(fil1, fil2, **initializer)
    xdivep = (xdivep1 + xdivep2 + xdivep3).compute()
    xdivep.name = 'Div of zonal EP flux'
    xdivep.math = (r"""-$\frac{1}{\bar{\sigma}}"""
                   r"""(\bar{\sigma}\widehat{u ^{\prime \prime} """
                   r"""v ^{\prime \prime}})_{\tilde{x}}$""")
    xdivep.units = r'ms$^{-2}$'
    return xdivep


def get_ydivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hvvym = ds1.hvv_Cv.ysm().yep().read().divide_by('dxCv').dbyd(
            2).nanmean(axis=0).move_to('v').compute()
        ydivep1 = (-hvvym / h).compute()
        return ydivep1


def get_ydivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    vr = get_vr(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        hvmy = ds2.vh.ysm().yep().read().nanmean(
            axis=0).divide_by('dxCv').dbyd(2).move_to('v').compute()
        ydivep3 = (hvmy * vr / h).compute()
        return ydivep3


def get_edlsqmy(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        esq = ds1.esq.yep().read().nanmean(axis=0)
        e = ds2.e.yep().zep().read().nanmean(axis=0).move_to('l')**2
        e = e.compute(check_loc=False)
        edlsqmy = esq - e
        edlsqmy = edlsqmy.dbyd(2).compute()
        return edlsqmy


def get_ydivep4(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    edlsqmy = get_edlsqmy(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        db = -np.diff(ds1.zl)[0] * 9.8 / 1031
        ydivep4 = (-0.5 * edlsqmy * db / h).compute()
        return ydivep4


def get_ydivep(fil1, fil2, **initializer):
    ydivep1 = get_ydivep1(fil1, fil2, **initializer)
    ydivep2 = -get_advy(fil1, fil2, **initializer)
    ydivep3 = get_ydivep3(fil1, fil2, **initializer)
    ydivep4 = get_ydivep4(fil1, fil2, **initializer)
    ydivep = ((ydivep1 + ydivep2).compute() +
              (ydivep3 + ydivep4).compute()).compute()
    ydivep.name = 'Div of merid EP flux'
    ydivep.math = (
        r"""-$\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
        r"""\widehat{v ^{\prime \prime} v ^{\prime \prime} })_{\tilde{y}}$"""
        r"""-$\frac{1}{2\bar{\sigma}}"""
        r"""(\bar{\zeta ^{\prime 2}})_{\tilde{y}}$""")
    ydivep.units = r'ms$^{-2}$'
    return ydivep


def get_bdivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hvwbm = ds1.twa_hvwb.read().nanmean(axis=0).compute()
        bdivep1 = (hvwbm / h).compute()
        return bdivep1


def get_bdivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    vr = get_vr(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        hwb = ds2.wd.yep().zep().read().nanmean(axis=0).move_to('v').np_ops(
            np.diff, axis=1, sets_vloc='l').compute()
        bdivep3 = (hwb * vr / h).compute()
        return bdivep3


def get_bdivep4(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    edlsqmy = get_edlsqmy(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        hpfv = ds1.twa_hpfv.read().nanmean(axis=0).compute()
        pfvm = ds2.PFv.read().nanmean(axis=0).compute()
        db = -np.diff(ds1.zl)[0] * 9.8 / 1031
        edpfvdmb = -(-hpfv + (h * pfvm).compute() -
                     (edlsqmy * db * 0.5).compute()) / h
        bdivep4 = edpfvdmb.compute()
        return bdivep4


def get_bdivep(fil1, fil2, **initializer):
    bdivep1 = get_bdivep1(fil1, fil2, **initializer)
    bdivep2 = -get_advb(fil1, fil2, **initializer)
    bdivep3 = get_bdivep3(fil1, fil2, **initializer)
    bdivep4 = get_bdivep4(fil1, fil2, **initializer)
    bdivep = (bdivep1 + bdivep2 + bdivep3 + bdivep4).compute()
    bdivep.name = 'Form Drag'
    bdivep.math = (r"""-$\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
                   r"""\widehat{v ^{\prime \prime} """
                   r"""\varpi ^{\prime \prime}})_{\tilde{b}}$"""
                   r""" -$\frac{1}{2\bar{\sigma}}"""
                   r"""(\overline{\zeta ^\prime m_{\tilde{y}}^\prime})"""
                   r"""_{\tilde{b}}$""")
    bdivep.units = r'ms$^{-2}$'
    return bdivep


def get_Y1twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdiffvm = ds1.twa_hdiffv.read().nanmean(axis=0).compute()
        Y1twa = (hdiffvm / h).compute()
        Y1twa.name = 'Horizonal diffusion'
        Y1twa.math = r'$\widehat{Y^H}$'
        Y1twa.units = r'ms$^{-2}$'
        return Y1twa


def get_Y2twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdvdtviscm = ds1.twa_hdvdtvisc.read().nanmean(axis=0).compute()
        Y2twa = (hdvdtviscm / h).compute()
        Y2twa.name = 'Veritical viscous forces'
        Y2twa.math = r'$\widehat{Y^V}$'
        Y2twa.units = r'ms$^{-2}$'
        return Y2twa


def get_ByminusPVflux(fil1, fil2, **initializer):
    advx = get_advx(fil1, fil2, **initializer)
    advy = get_advy(fil1, fil2, **initializer)
    cor = get_cor(fil1, fil2, **initializer)
    pfvm = get_pfvm(fil1, fil2, **initializer)
    ByminusPVflux = (advx + advy + cor + pfvm).compute()
    return ByminusPVflux


def get_PV(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    htol = initializer.get('htol', 1e-3)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_fory = ds1.h_Cu.xsm().yep().read().nanmean(axis=0).compute(
            check_loc=False)
        a = h_fory.values
        a[a < htol] = np.nan
        h_fory.values = a
        ury = ds2.uh.xsm().yep().read().nanmean(
            axis=0).divide_by('dyCu') / h_fory
        ury = ury.dbyd(2)
        f_slice = ury.get_slice_2D()._slice_2D
        ury = ury.move_to('v').compute()
        h_forx = ds1.h_Cv.xep().xsm().read().nanmean(axis=0).compute()
        a = h_forx.values
        a[a < htol] = np.nan
        h_forx.values = a
        vrx = ds2.vh.xep().xsm().read().nanmean(
            axis=0).divide_by('dxCv') / h_forx
        vrx = vrx.dbyd(3).move_to('v').compute()
        f = initializer['geometry'].f[f_slice]
        f = 0.5 * (f[:, :-1] + f[:, 1:])
        PV = ((vrx - ury + f) / h).compute()
        PV.name = r'PV$^\sharp$'
        return PV


def get_PVflux(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    PV = get_PV(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        uh = ds2.uh.xsm().yep().read().divide_by('dyCu').nanmean(
            axis=0).move_to('h').move_to('v').compute()
        PVflux = (-PV * uh).compute()
        PVflux.name = 'PV flux'
        PVflux.math = r'$-\bar{\sigma}\hat{u}\Pi^{\sharp}$'
        PVflux.units = r'ms$^{-2}$'
        return PVflux


def get_By(fil1, fil2, **initializer):
    PVflux = get_PVflux(fil1, fil2, **initializer)
    ByminusPVflux = get_ByminusPVflux(fil1, fil2, **initializer)
    By = (ByminusPVflux - PVflux).compute()
    By.name = 'Grad Bernoulli func'
    By.math = r'$-\bar{B}_{\tilde{y}}$'
    By.units = r'ms$^{-2}$'
    return By


def extract_twamomy_terms(fil1, fil2, **initializer):

    conventional_list = [
        get_advx, get_advy, get_advb, get_cor, get_pfvm, get_xdivep,
        get_ydivep, get_bdivep, get_Y1twa, get_Y2twa
    ]

    withPVflux_list = [
        get_PVflux, get_advb, get_By, get_xdivep, get_ydivep, get_bdivep,
        get_Y1twa, get_Y2twa
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
    initializer['final_loc'] = 'vi'
    with pym6.Dataset(fil2, **initializer) as ds:
        e = ds.e.yep().zep().read().move_to('v').nanmean(axis=(0, 2)).compute()
    initializer['final_loc'] = 'vl'
    blist = extract_twamomy_terms(fil1, fil2, **initializer)
    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.nanmean(axis=meanax)
        if z is not None:
            b = b.toz(z, e)
        blist[i] = b.to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Meridional Momentum Budget'
    return blist_concat, blist


def plot_budget(fil1, fil2, **initializer):
    blist_concat, blist = extract_budget(fil1, fil2, **initializer)
    fg = blist_concat.plot.imshow(
        'xh',
        'z',
        size=2,
        aspect=(1 + np.sqrt(5)) / 2,
        yincrease=True,
        vmin=-5e-6,
        vmax=5e-6,
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
    extract_twamomy_terms(fil, initializer)
