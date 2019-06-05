import pymom6.pymom6 as pym6
import numpy as np
import importlib
import xarray as xr
import pandas as pd
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
    bc = dict(
        v=['mirror', 'circsymh', 'zeros', 'circsymq', 'circsymh', 'mirror'])
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_forx = ds1.h_Cv.xsm().xep().bc_type(bc).read().nanmean(
            axis=0).compute()
        a = h_forx.values
        a[a < htol] = np.nan
        h_forx.values = a
        vrx = ds2.vh.xsm().xep().read().nanmean(axis=0) / h_forx
        vrx = vrx.compute(check_loc=False)
        a = vrx.values
        a[np.isnan(h_forx.values)] = 0
        vrx.values = a
        vrx = vrx.dbyd(3, weights='area').move_to('v').compute()
        hum = ds2.uh.fillvalue(0).yep().xsm().read().nanmean(
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
        vry = ds2.vh.ysm().yep().read().nanmean(axis=0) / h_fory
        vry = vry.dbyd(2, weights='area').move_to('v').compute()
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
        db = np.diff(ds2.zl)[0] * 9.8 / 1000
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
        # pfvm = h * pfvm / h
        # pfvm = pfvm.compute()
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
        hvvym = ds1.hvv_Cv.ysm().yep().read().dbyd(
            2, weights='area').nanmean(axis=0).move_to('v').compute()
        huvxm = huvxphvvym + hvvym
        xdivep1 = (huvxm / h).compute()
    return xdivep1


def get_xdivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    vr = get_vr(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        humx = ds2.uh.fillvalue(0).yep().xsm().read().nanmean(axis=0).dbyd(
            3, weights='area').move_to('v').compute()
        xdivep3 = (humx * vr / h).compute()
    return xdivep3


def get_xdivep(fil1, fil2, **initializer):
    xdivep1 = get_xdivep1(fil1, fil2, **initializer)
    xdivep2 = -get_advx(fil1, fil2, **initializer)
    xdivep3 = get_xdivep3(fil1, fil2, **initializer)
    xdivep = (xdivep1 + xdivep2.compute() + xdivep3).compute()
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
        hvvym = ds1.hvv_Cv.ysm().yep().read().dbyd(
            2, weights='area').nanmean(axis=0).move_to('v').compute()
        ydivep1 = (-hvvym / h).compute()
    return ydivep1


def get_ydivep3(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    vr = get_vr(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        hvmy = ds2.vh.ysm().yep().read().nanmean(axis=0).dbyd(
            2, weights='area').move_to('v').compute()
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
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
        ydivep4 = ((-0.5 * db * edlsqmy).compute() / h).compute()
    ydivep4.name = 'Merid grad of EPE'
    ydivep4.math = (r"""-$\frac{1}{2\bar{\sigma}}"""
                    r"""(\bar{\zeta ^{\prime 2}})_{\tilde{y}}$""")
    return ydivep4


def get_ydivep(fil1, fil2, **initializer):
    ydivep1 = get_ydivep1(fil1, fil2, **initializer)
    ydivep2 = -get_advy(fil1, fil2, **initializer)
    ydivep3 = get_ydivep3(fil1, fil2, **initializer)
    ydivep4 = get_ydivep4(fil1, fil2, **initializer)
    ydivep = ((ydivep1 + ydivep2.compute()).compute() +
              (ydivep3 + ydivep4).compute()).compute()
    ydivep.name = 'Div of merid EP flux'
    ydivep.math = (
        r"""-$\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
        r"""\widehat{v ^{\prime \prime} v ^{\prime \prime} })_{\tilde{y}}$"""
        r"""-$\frac{1}{2\bar{\sigma}}"""
        r"""(\bar{\zeta ^{\prime 2}})_{\tilde{y}}$""")
    ydivep.units = r'ms$^{-2}$'
    return ydivep


def get_ydivRS(fil1, fil2, **initializer):
    ydivep1 = get_ydivep1(fil1, fil2, **initializer)
    ydivep2 = -get_advy(fil1, fil2, **initializer)
    ydivep3 = get_ydivep3(fil1, fil2, **initializer)
    ydivRS = ((ydivep1 + ydivep2.compute()).compute() +
              (ydivep3).compute()).compute()
    ydivRS.name = 'Div of merid RS'
    ydivRS.math = (
        r"""-$\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
        r"""\widehat{v ^{\prime \prime} v ^{\prime \prime} })_{\tilde{y}}$""")

    ydivRS.units = r'ms$^{-2}$'
    return ydivRS


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
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
        edpfvdmb = -((-hpfv).compute() + (h * pfvm).compute() -
                     (edlsqmy * db * 0.5).compute()) / h
        bdivep4 = edpfvdmb.compute()
    bdivep4.name = 'Form drag'
    bdivep4.math = (r""" -$\frac{1}{\bar{\sigma}}"""
                    r"""(\overline{\zeta ^\prime m_{\tilde{y}}^\prime})"""
                    r"""_{\tilde{b}}$""")
    bdivep4.units = r'ms$^{-2}$'
    return bdivep4


def get_bdivep(fil1, fil2, **initializer):
    bdivep1 = get_bdivep1(fil1, fil2, **initializer)
    bdivep2 = -get_advb(fil1, fil2, **initializer)
    bdivep3 = get_bdivep3(fil1, fil2, **initializer)
    bdivep4 = get_bdivep4(fil1, fil2, **initializer)
    bdivep = (bdivep1 + bdivep2.compute() + bdivep3 + bdivep4).compute()
    bdivep.name = 'Form Drag'
    bdivep.math = (r"""-$\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
                   r"""\widehat{v ^{\prime \prime} """
                   r"""\varpi ^{\prime \prime}})_{\tilde{b}}$"""
                   r""" -$\frac{1}{2\bar{\sigma}}"""
                   r"""(\overline{\zeta ^\prime m_{\tilde{y}}^\prime})"""
                   r"""_{\tilde{b}}$""")
    bdivep.units = r'ms$^{-2}$'
    return bdivep


def get_bdivRS(fil1, fil2, **initializer):
    bdivep1 = get_bdivep1(fil1, fil2, **initializer)
    bdivep2 = -get_advb(fil1, fil2, **initializer)
    bdivep3 = get_bdivep3(fil1, fil2, **initializer)
    bdivRS = (bdivep1 + bdivep2.compute() + bdivep3).compute()
    bdivRS.name = 'Vertical grad RS'
    bdivRS.math = (r"""-$\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
                   r"""\widehat{v ^{\prime \prime} """
                   r"""\varpi ^{\prime \prime}})_{\tilde{b}}$""")
    bdivRS.units = r'ms$^{-2}$'
    return bdivRS


def get_divep(fil1, fil2, **initializer):
    xdivep = get_xdivep(fil1, fil2, **initializer)
    ydivep = get_ydivep(fil1, fil2, **initializer)
    bdivep = get_bdivep(fil1, fil2, **initializer)
    divep = (xdivep + ydivep + bdivep).compute()
    divep.math = r"$-\vec{\nabla} \cdot \vec{E^v}$"
    divep.units = r'ms$^{-2}$'
    divep.name = r'Div EP flux'
    return divep


def get_Y1twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdiffvm = ds1.twa_hdiffv.read().nanmean(axis=0).compute()
        Y1twa = (hdiffvm / h).compute()
    Y1twa.name = 'Horizonal friction'
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
        ury = ds2.uh.xsm().yep().read().nanmean(axis=0) / h_fory
        ury = ury.compute(check_loc=False)
        a = ury.values
        a[np.isnan(h_fory.values)] = 0
        ury.values = a
        ury = ury.dbyd(2, weights='area')
        f_slice = ury.get_slice_2D()._slice_2D
        ury = ury.move_to('v').compute()
        h_forx = ds1.h_Cv.xep().xsm().read().nanmean(axis=0).compute()
        a = h_forx.values
        a[a < htol] = np.nan
        h_forx.values = a
        vrx = ds2.vh.xep().xsm().read().nanmean(axis=0) / h_forx
        vrx = vrx.compute(check_loc=False)
        a = vrx.values
        a[np.isnan(h_forx.values)] = 0
        vrx.values = a
        vrx = vrx.dbyd(3, weights='area').move_to('v').compute()
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


def get_uvflux(fil1, fil2, **initializer):
    h = get_h(fil1, fil2, **initializer)
    vr = get_vr(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds, pym6.Dataset(
            fil2, **initializer) as ds2:
        hum = ds2.uh.final_loc('vl').yep().xsm().read().nanmean(
            axis=0).divide_by('dyCu').move_to('h').move_to('v').compute()
        huvm = ds.huv_Bu.final_loc('vl').xsm().read().nanmean(
            axis=0).move_to('v').compute()
        uv_e_flux = ((huvm - (vr * hum).compute()) / h).compute()
        ex = (ds2.e.final_loc('vl').zep().yep().xsm().xep().read().nanmean(
            axis=0).dbyd(3).move_to('h').move_to('v').move_to('l').compute())
        uv_e_flux_ex = (ex * uv_e_flux).compute()
    uv_e_flux.math = r"""$\widehat{u^{\prime \prime}v^{\prime \prime}}$"""
    uv_e_flux.name = r"Zonal comp of $\vec{E^v}$"
    uv_e_flux_ex.math = r"""$\widehat{u^{\prime \prime}v^{\prime \prime}}$"""
    uv_e_flux_ex.name = r"Vert comp of $\vec{E^v}$"
    return uv_e_flux, uv_e_flux_ex


def get_fdflux(fil1, fil2, **initializer):
    fd = get_bdivep4(fil1, fil2, **initializer)
    h = get_h(fil1, fil2, **initializer)
    fd = (fd * h).compute()
    fd.values = -np.cumsum((fd.values)[:, ::-1], axis=1)[:, ::-1]
    fd.math = r"""$\overline{\zeta^\prime m_{\tilde{y}}^\prime}$"""
    fd.name = r"Vert comp of $\vec{E^v}$"
    return fd


def get_epe_flux(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
        esq = ds1.esq.yep().read().nanmean(axis=0)
        e = ds2.e.yep().zep().read().nanmean(axis=0).move_to('l')**2
        e = e.compute(check_loc=False)
        edlsqm = esq - e
        edlsqm = edlsqm.move_to('v').compute()
        epeflux = (edlsqm / h * 0.5 * db).compute()
        ex = (ds2.e.final_loc('vl').zep().yep().xsm().xep().read().nanmean(
            axis=0).dbyd(3).move_to('h').move_to('v').move_to('l').compute())
        epeflux_ex = (ex * epeflux).compute()
    return epeflux, epeflux_ex


def get_eprimesq(fil1, fil2, **initializer):
    #initializer['final_loc'] = 'vl'
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        esq = ds1.esq.read().nanmean(axis=0)
        e = ds2.e.zep().read().nanmean(axis=0).move_to('l')**2
        e = e.compute(check_loc=False)
        edlsqm = esq - e
        edlsqm = edlsqm.compute()
    return edlsqm


def get_pvfluxduetouv(fil1, fil2, **initializer):
    uv = -get_xdivep(fil1, fil2, **initializer)
    uv = uv.compute()
    with pym6.Dataset(fil2, **initializer) as ds:
        ex = ds.e.final_loc('vl').zep().xsm().xep().yep().read().dbyd(
            3).move_to('h').move_to('v').move_to('l').nanmean(
                axis=0).compute()
    uvex = (ex * uv).compute()
    return uv, uvex


def get_pvfluxduetofd(fil1, fil2, **initializer):
    fd = -get_bdivep4(fil1, fil2, **initializer)
    fd = fd.compute()
    with pym6.Dataset(fil2, **initializer) as ds:
        ex = ds.e.final_loc('vl').zep().xsm().xep().yep().read().dbyd(
            3).move_to('h').move_to('v').move_to('l').nanmean(
                axis=0).compute()
    fdex = (ex * fd).compute()
    return fd, fdex


def get_pvfluxduetoY(fil1, fil2, **initializer):
    Y = -get_Y1twa(fil1, fil2, **initializer)
    Y = Y.compute()
    with pym6.Dataset(fil2, **initializer) as ds:
        ex = ds.e.final_loc('vl').zep().xsm().xep().yep().read().dbyd(
            3).move_to('h').move_to('v').move_to('l').nanmean(
                axis=0).compute()
    Yex = (ex * Y).compute()
    return Y, Yex


def extract_twamomy_terms(fil1, fil2, **initializer):

    conventional_list = [
        get_advx, get_advy, get_advb, get_cor, get_pfvm, get_xdivep,
        get_ydivep, get_bdivep4, get_Y1twa, get_Y2twa
    ]

    withPVflux_list = [
        get_PVflux, get_advb, get_By, get_xdivep, get_ydivep, get_bdivep,
        get_Y1twa, get_Y2twa
    ]

    forPV_list = [
        get_PVflux, get_advb, get_By, get_xdivep, get_ydivRS, get_bdivRS,
        get_Y1twa, get_Y2twa, get_ydivep4, get_bdivep4
    ]

    combineEPflux_list = [
        get_advx, get_advy, get_advb, get_cor, get_pfvm, get_divep, get_Y1twa,
        get_Y2twa
    ]
    onlyEPfluxes_list = [get_xdivep, get_bdivep4]

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
    elif type == 'forPV':
        for i, func in enumerate(forPV_list):
            return_list.append(func(fil1, fil2, **initializer))
    elif type == 'combineEPflux':
        for i, func in enumerate(combineEPflux_list):
            if i in only:
                return_list.append(func(fil1, fil2, **initializer))
    elif type == 'divEPflux':
        for i, func in enumerate(onlyEPfluxes_list):
            if i in only:
                return_list.append(func(fil1, fil2, **initializer))
    return return_list


def extract_budget(fil1, fil2, fil3=None, **initializer):
    meanax = initializer.get('meanax', 2)
    initializer.pop('meanax')
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'vi'
    with pym6.Dataset(fil2, **initializer) as ds:
        e = ds.e.yep().read().move_to('v').nanmean((0, meanax)).compute()
    initializer['final_loc'] = 'vl'
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
            swash = swash.xsm().read().move_to('v').nanmean(
                (0, meanax)).compute()
            swash = ((-swash + islaydeepmax) / islaydeepmax * 100).compute()
            if z is not None:
                swash = swash.toz(z, e)
            if initializer.get('tokm', True):
                if meanax == 2:
                    swash = swash.tokm(3)
                elif meanax == 3:
                    swash = swash.tokm(2)
            swash = swash.to_DataArray()
    else:
        swash = None
    blist = extract_twamomy_terms(fil1, fil2, **initializer)
    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.nanmean(axis=meanax)
        if z is not None:
            b = b.toz(z, e)
        if initializer.get('tokm', True):
            if meanax == 2:
                b = b.tokm(3)
            elif meanax == 3:
                b = b.tokm(2)
        blist[i] = b.to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Merid momentum budget'
    if initializer.get('tokm', True):
        if meanax == 2:
            e = e.tokm(3)
        elif meanax == 3:
            e = e.tokm(2)
    e = e.to_DataArray()
    return_dict = dict(
        blist_concat=blist_concat, blist=blist, e=e, swash=swash)
    return return_dict
