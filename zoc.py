import pymom6.pymom6 as pym6
import numpy as np
import importlib
import xarray as xr
import pandas as pd
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
        ur.values[np.isnan(ur.values)] = 0
    return ur


def get_advx(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_forx = ds1.h_Cu.xsm().xep().read().nanmean(axis=0).compute()
        a = h_forx.values
        a[a < htol] = np.nan
        h_forx.values = a
        urx = ds2.uh.xsm().xep().read().nanmean(axis=0) / h_forx
        urx = urx.dbyd(3, weights='area').move_to('u').compute()
        urx.values[np.isnan(urx.values)] = 0
        uh = ds2.uh.read().nanmean(axis=0).divide_by('dyCu').compute()
        advx = (-urx * uh).multiply_by('dyCu').reduce_(
            np.nansum, axis=2).compute()
    advx.name = 'Zonal advection'
    advx.math = r'$-\bar{\sigma}\hat{u}\hat{u}_{\tilde{x}}$'
    advx.units = r'm$^2$s$^{-2}$'
    return advx


def get_advy(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_fory = ds1.h_Cu.ysm().yep().read().nanmean(axis=0).compute()
        a = h_fory.values
        a[a < htol] = np.nan
        h_fory.values = a
        ury = ds2.uh.ysm().yep().read().nanmean(axis=0) / h_fory
        ury = ury.dbyd(2, weights='area').move_to('u').compute()
        ury.values[np.isnan(ury.values)] = 0
        hvm = ds2.vh.ysm().xep().read().nanmean(
            axis=0).divide_by('dxCv').move_to('h').move_to('u').compute()
        advy = (-hvm * ury).multiply_by('dyCu').reduce_(
            np.nansum, axis=2).compute()
    advy.name = 'Meridional advection'
    advy.math = r'$-\bar{\sigma}\hat{v}\hat{u}_{\tilde{y}}$'
    advy.units = r'm$^2$s$^{-2}$'
    return advy


def get_advb(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_forb = ds1.h_Cu.zsm().zep().read().nanmean(axis=0).compute()
        a = h_forb.values
        a[a < htol] = np.nan
        h_forb.values = a
        urb = ds2.uh.zsm().zep().read().nanmean(axis=0) / h_forb
        urb = urb.divide_by('dyCu').dbyd(1).move_to('l').compute()
        urb.values[np.isnan(urb.values)] = 0
        db = np.diff(ds2.zl)[0] * 9.8 / 1000
        hwm = ds2.wd.xep().zep().read().nanmean(
            axis=0).move_to('l').move_to('u') * db
        advb = (-hwm * urb).multiply_by('dyCu').reduce_(
            np.nansum, axis=2).compute()
    advb.name = 'Vertical advection'
    advb.math = r'$-\bar{\sigma}\hat{\varpi}\hat{u}_{\tilde{b}}$'
    advb.units = r'm$^2$s$^{-2}$'
    return advb


def get_cor(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hfvm = ds1.twa_hfv.read().nanmean(axis=0).compute()
        cor = hfvm.multiply_by('dyCu').reduce_(np.nansum, axis=2).compute()
    cor.name = 'Coriolis force'
    cor.math = r'$f\bar{\sigma}\hat{v}$'
    cor.units = r'm$^2$s$^{-2}$'
    return cor


def get_pfum(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        pfum = ds2.PFu.read().nanmean(axis=0).compute()
        pfum = h * pfum
        pfum = pfum.compute()
        pfum.values[np.isnan(pfum.values)] = 0
        pfum = pfum.multiply_by('dyCu').reduce_(np.nansum, axis=2).compute()
    pfum.name = 'Grad of Montg Pot'
    pfum.math = r'$-\bar{\sigma}\bar{m}_{\tilde{x}}$'
    pfum.units = r'm$^2$s$^{-2}$'
    return pfum


def get_corpfum(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        hfvm = ds1.twa_hfv.read().nanmean(axis=0).compute()
        pfum = ds2.PFu.read().nanmean(axis=0).compute()
        pfum = (h * pfum).compute()
        corpfum = (hfvm + pfum).multiply_by('dyCu').reduce_(
            np.nansum, axis=2).compute()
    corpfum.name = 'Ageostrophic term'
    corpfum.math = r'$f\bar{\sigma}\hat{v}-\bar{\sigma}\bar{m}_{\tilde{x}}$'
    corpfum.units = r'm$^2$s$^{-2}$'
    return corpfum


def get_xdivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    with pym6.Dataset(fil1, **initializer) as ds1:
        huuxm = ds1.huu_Cu.xsm().xep().read().dbyd(
            3, weights='area').nanmean(axis=0).move_to('u').compute()
        xdivep1 = (
            -huuxm.multiply_by('dyCu').reduce_(np.nansum, axis=2)).compute()
    return xdivep1


def get_xdivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    ur = get_ur(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        humx = ds2.uh.xsm().xep().read().nanmean(axis=0).dbyd(
            3, weights='area').move_to('u').compute()
        xdivep3 = (humx * ur).multiply_by('dyCu').reduce_(
            np.nansum, axis=2).compute()
    return xdivep3


def get_edlsqmx(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        esq = ds1.esq.xep().read().nanmean(axis=0)
        e = ds2.e.xep().zep().read().nanmean(axis=0).move_to('l')**2
        e = e.compute(check_loc=False)
        edlsqmx = esq - e
        edlsqmx = edlsqmx.dbyd(3).multiply_by('dyCu').reduce_(
            np.nansum, axis=2).compute()
    return edlsqmx


def get_xdivep4(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    edlsqmx = get_edlsqmx(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
        xdivep4 = ((-0.5 * db * edlsqmx).compute()).compute()
        xdivep4.math = (r"""-$\frac{1}{2}"""
                        r"""(\bar{\zeta ^{\prime 2}})_{\tilde{x}}$""")
    xdivep4.units = r'm$^2$s$^{-2}$'
    return xdivep4


def get_xdivep(fil1, fil2, **initializer):
    xdivep1 = get_xdivep1(fil1, fil2, **initializer)
    xdivep2 = -get_advx(fil1, fil2, **initializer)
    xdivep3 = get_xdivep3(fil1, fil2, **initializer)
    xdivep4 = get_xdivep4(fil1, fil2, **initializer)
    xdivep = (xdivep1 + xdivep2.compute() + xdivep3 + xdivep4).compute()
    xdivep.name = 'Div of zonal EP flux'
    xdivep.math = (
        r"""-$(\bar{\sigma}"""
        r"""\widehat{u ^{\prime \prime} u ^{\prime \prime} })_{\tilde{x}}"""
        r"""-\frac{1}{2}"""
        r"""(\bar{\zeta ^{\prime 2}})_{\tilde{x}}$""")
    xdivep.units = r'm$^2$s$^{-2}$'
    return xdivep


def get_xdivRS(fil1, fil2, **initializer):
    xdivep1 = get_xdivep1(fil1, fil2, **initializer)
    xdivep2 = -get_advx(fil1, fil2, **initializer)
    xdivep3 = get_xdivep3(fil1, fil2, **initializer)
    xdivRS = (xdivep1 + xdivep2.compute() + xdivep3).compute()
    xdivRS.name = 'Zonal grad RS'
    xdivRS.math = (
        r"""-$(\bar{\sigma}"""
        r"""\widehat{u ^{\prime \prime} u ^{\prime \prime} })_{\tilde{x}}$""")
    xdivRS.units = r'm$^2$s$^{-2}$'
    return xdivRS


def get_ydivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    with pym6.Dataset(fil1, **initializer) as ds1:
        huuxpt = ds1.twa_huuxpt.read().nanmean(axis=0).compute()
        huvymt = ds1.twa_huvymt.read().nanmean(axis=0).compute()
        huuxphuvym = huuxpt + huvymt
        huuxphuvym = huuxphuvym.compute()
        huuxm = ds1.huu_Cu.xsm().xep().read().dbyd(
            3, weights='area').nanmean(axis=0).move_to('u').compute()
        huvym = huuxphuvym + huuxm
        ydivep1 = (huvym.multiply_by('dyCu').reduce_(np.nansum,
                                                     axis=2)).compute()
    return ydivep1


def get_ydivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    ur = get_ur(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        hvmy = ds2.vh.ysm().xep().read().nanmean(axis=0).dbyd(
            2, weights='area').move_to('u').compute()
        ydivep3 = (hvmy * ur).multiply_by('dyCu').reduce_(
            np.nansum, axis=2).compute()
    return ydivep3


def get_ydivep(fil1, fil2, **initializer):
    ydivep1 = get_ydivep1(fil1, fil2, **initializer)
    ydivep2 = -get_advy(fil1, fil2, **initializer)
    ydivep3 = get_ydivep3(fil1, fil2, **initializer)
    ydivep = (ydivep1 + ydivep2.compute() + ydivep3).compute()
    ydivep.name = 'Div of merid EP flux'
    ydivep.math = (r"""-$(\bar{\sigma}\widehat{u ^{\prime \prime} """
                   r"""v ^{\prime \prime}})_{\tilde{y}}$""")
    ydivep.units = r'm$^2$s$^{-2}$'
    return ydivep


def get_bdivep1(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    with pym6.Dataset(fil1, **initializer) as ds1:
        huwbm = ds1.twa_huwb.read().nanmean(axis=0).compute()
        bdivep1 = (huwbm.multiply_by('dyCu').reduce_(np.nansum,
                                                     axis=2)).compute()
    return bdivep1


def get_bdivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    ur = get_ur(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        hwb = ds2.wd.xep().zep().read().nanmean(axis=0).move_to('u').np_ops(
            np.diff, axis=1, sets_vloc='l').compute()
        bdivep3 = (hwb * ur).multiply_by('dyCu').reduce_(
            np.nansum, axis=2).compute()
    return bdivep3


def get_bdivep4(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    edlsqmx = get_edlsqmx(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    h.values[np.isnan(h.values)] = 0
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        hpfu = ds1.twa_hpfu.read().nanmean(axis=0).compute()
        pfum = ds2.PFu.read().nanmean(axis=0).compute()
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
        edpfudmb = -((-hpfu).compute() +
                     (h * pfum).compute()).multiply_by('dyCu').reduce_(
                         np.nansum, axis=2).compute()
        bdivep4 = (edpfudmb + (edlsqmx * db * 0.5).compute()).compute()
    bdivep4.name = 'Form drag'
    bdivep4.math = (r"""-$(\overline{\zeta ^\prime m_{\tilde{x}}^\prime})"""
                    r"""_{\tilde{b}}$""")
    bdivep4.units = r'm$^2$s$^{-2}$'
    return bdivep4


def get_bdivep(fil1, fil2, **initializer):
    bdivep1 = get_bdivep1(fil1, fil2, **initializer)
    bdivep2 = -get_advb(fil1, fil2, **initializer)
    bdivep3 = get_bdivep3(fil1, fil2, **initializer)
    bdivep4 = get_bdivep4(fil1, fil2, **initializer)
    bdivep = (bdivep1 + bdivep2.compute() + bdivep3 + bdivep4).compute()
    bdivep.name = 'Vert div of EP flux'
    bdivep.math = (r"""-$(\bar{\sigma}"""
                   r"""\widehat{u ^{\prime \prime} """
                   r"""\varpi ^{\prime \prime}})_{\tilde{b}}$"""
                   r"""-$(\overline{\zeta ^\prime m_{\tilde{x}}^\prime})"""
                   r"""_{\tilde{b}}$""")
    bdivep.units = r'm$^2$s$^{-2}$'
    return bdivep


def get_bdivRS(fil1, fil2, **initializer):
    bdivep1 = get_bdivep1(fil1, fil2, **initializer)
    bdivep2 = -get_advb(fil1, fil2, **initializer)
    bdivep3 = get_bdivep3(fil1, fil2, **initializer)
    bdivRS = (bdivep1 + bdivep2 + bdivep3).compute()
    bdivRS.name = 'Vertical grad RS'
    bdivRS.math = (r"""-$(\bar{\sigma}"""
                   r"""\widehat{u ^{\prime \prime} """
                   r"""\varpi ^{\prime \prime}})_{\tilde{b}}$""")
    bdivRS.units = r'm$^2$s$^{-2}$'
    return bdivRS


def get_X1twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdiffum = ds1.twa_hdiffu.read().nanmean(axis=0).compute()
        X1twa = (hdiffum.multiply_by('dyCu').reduce_(np.nansum,
                                                     axis=2)).compute()
        X1twa.name = 'Horizontal friction'
        X1twa.math = r'$\bar{\sigma}\widehat{X^H}$'
        X1twa.units = r'm$^2$s$^{-2}$'
    return X1twa


def get_X2twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdudtviscm = ds1.twa_hdudtvisc.read().nanmean(axis=0).compute()
        X2twa = (hdudtviscm.multiply_by('dyCu').reduce_(np.nansum,
                                                        axis=2)).compute()
        X2twa.name = 'Vertical viscous forces'
        X2twa.math = r'$\bar{\sigma}\widehat{X^V}$'
        X2twa.units = r'm$^2$s$^{-2}$'
    return X2twa


def extract_twamomx_terms(fil1, fil2, **initializer):

    conventional_list = [
        get_advx, get_advy, get_advb, get_corpfum, get_xdivRS, get_xdivep4,
        get_ydivep, get_bdivRS, get_bdivep4, get_X1twa, get_X2twa
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
    return return_list


def extract_budget(fil1, fil2, fil3=None, **initializer):
    meanax = initializer.get('meanax', 2)
    initializer.pop('meanax')
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'ui'
    with pym6.Dataset(fil2, **initializer) as ds:
        e = ds.e.xep().zep().read().move_to('u').nanmean(axis=(0, 2)).compute()
    initializer['final_loc'] = 'ul'
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
            swash = swash.ysm().read().move_to('u').nanmean(
                axis=(0, 2)).compute()
            swash = ((-swash + islaydeepmax) / islaydeepmax * 100).compute()
            if z is not None:
                swash = swash.toz(z, e)
            swash = swash.tokm(3).to_DataArray()
    else:
        swash = None
    blist = extract_twamomx_terms(fil1, fil2, **initializer)
    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.nanmean(axis=meanax)
        if z is not None:
            b = b.toz(z, e)
        blist[i] = b.tokm(3).to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Zonal Momentum Budget'
    e = e.tokm(3).to_DataArray()
    return_dict = dict(
        blist_concat=blist_concat, blist=blist, e=e, swash=swash)
    return return_dict


def extract_momx_terms(fil1, fil2, **initializer):

    with pym6.Dataset(fil2, **initializer) as ds2:
        cau = ds2.CAu.read().nanmean(axis=0).compute()
        cau.name = 'Coriolis term'
        cau.math = r'$-f\bar{v}$'

        pfu = ds2.PFu.read().nanmean(axis=0).compute()
        pfu.name = 'Grad Mont Pot'
        pfu.math = r'$-\bar{m_{\tilde{x}}}$'

        dudt_dia = ds2.dudt_dia.read().nanmean(axis=0).compute()
        dudt_dia.name = 'Diapycnal advection'
        dudt_dia.math = r'$-\bar{\varpi} \bar{u}_{\tilde{b}}$'

        dudt_visc = ds2.du_dt_visc.read().nanmean(axis=0).compute()
        diffu = ds2.diffu.read().nanmean(axis=0).compute()
        diffu = (diffu + dudt_visc).compute()
        diffu.name = 'Friction terms'
        diffu.math = r'$\bar{X}$'

    conventional_list = [cau, pfu, dudt_dia, diffu]

    only = initializer.get('only', range(len(conventional_list)))
    if 'only' in initializer:
        return_list = []
        initializer.pop('only')
        for i, term in enumerate(conventional_list):
            if i in only:
                return_list.append(term)
    else:
        return_list = conventional_list
    return return_list
