import pymom6.pymom6 as pym6
import importlib
import numpy as np
import twamomx_budget as tx
import twamomy_budget as ty
importlib.reload(pym6)


def eke_conv(fil1, fil2, **kwargs):
    with pym6.Dataset(fil1, **kwargs) as ds1, pym6.Dataset(fil2,
                                                           **kwargs) as ds2:
        usqbar = ds1.usq.xsm().read().move_to('h').nanmean(axis=0).compute()
        ubarsq = ds1.u_masked.xsm().read().nanmean(axis=0).move_to('h')**2
        ubarsq = ubarsq.compute()
        vsqbar = ds1.vsq.ysm().read().move_to('h').nanmean(axis=0).compute()
        vbarsq = ds1.v_masked.ysm().read().move_to('h').nanmean(axis=0)**2
        vbarsq = vbarsq.compute()
        eke = 0.5 * ((usqbar - ubarsq).compute() +
                     (vsqbar - vbarsq).compute()).compute()
        h = ds2.e.zep().read().nanmean(axis=0).np_ops(
            np.diff, axis=1, sets_vloc='l').compute()
        h = (-h).compute()
        eke = (eke * h).compute()
    eke.name = r'EKE (m$^2$s$^{-2}$)'
    return eke


def eke(fil1, fil2, **kwargs):
    htol = kwargs.get('htol', 1e-10)
    with pym6.Dataset(fil1, **kwargs) as ds1, pym6.Dataset(fil2,
                                                           **kwargs) as ds2:
        huu = ds1.huu_Cu.xsm().read().nanmean(
            axis=0).divide_by('dyCu').compute(check_loc=False)
        uh = ds2.uh.xsm().read().nanmean(axis=0).divide_by('dyCu').compute(
            check_loc=False)
        ur = ds2.uh.xsm().read().nanmean(axis=0).compute(check_loc=False)
        h = ds1.h_Cu.xsm().read().nanmean(axis=0).compute(check_loc=False)
        ur = (ur / h).compute(check_loc=False)
        ur.values[h.values < htol] = 0
        ur = ur.divide_by('dyCu').compute(check_loc=False)
        ueke = (
            huu - (uh * ur).compute(check_loc=False)).move_to('h').compute()

        hvv = ds1.hvv_Cv.ysm().read().nanmean(
            axis=0).divide_by('dxCv').compute(check_loc=False)
        vh = ds2.vh.ysm().read().nanmean(axis=0).divide_by('dxCv').compute(
            check_loc=False)
        vr = ds2.vh.ysm().read().nanmean(axis=0).compute(check_loc=False)
        h = ds1.h_Cv.ysm().read().nanmean(axis=0).compute(check_loc=False)
        vr = (vr / h).compute(check_loc=False)
        vr.values[h.values < htol] = 0
        vr = vr.divide_by('dxCv').compute(check_loc=False)
        veke = (
            hvv - (vh * vr).compute(check_loc=False)).move_to('h').compute()

        eke = (0.5 * (ueke + veke)).compute()
    eke.name = r'EKE (m$^3$s$^{-2}$)'
    return eke


def mape(fil1, fil2):
    htol = kwargs.get('htol', 1e-10)
    with pym6.Dataset(fil1) as ds1, pym6.Dataset(fil2) as ds2:
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
        e = ds2.e.final_loc('hl').zep().read().nanmean(
            axis=0).move_to('h').compute()
        espatialmean = ds2.e.final_loc('hl').zep().read().nanmean(
            axis=(0, 2, 3)).movw_to('h').compute()
        mape = ((e * e).compute().nanmean(axis=(2, 3)).compute() -
                (espatialmean**2).compute()).compute()
        mape = (mape * db * 0.5).compute()
    mape.name = r'MAPE (m$^3$s$^{-2}$)'
    return mape


def eape(fil1, fil2, **kwargs):
    htol = kwargs.get('htol', 1e-10)
    # meanape = mape(fil1, fil2, **kwargs)
    with pym6.Dataset(fil1, **kwargs) as ds1, pym6.Dataset(fil2,
                                                           **kwargs) as ds2:
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
        esqm = ds1.esq.read().nanmean(axis=0).compute()
        emsq = ds2.e.final_loc('hl').zep().read().nanmean(
            axis=0).move_to('l')**2
        emsq = emsq.compute()
        eape = ((esqm - emsq) * db * 0.5).compute()
    eape.name = r'EAPE (m$^3$s$^{-2}$)'
    return eape


def sigum(fil0, **kwargs):
    with pym6.Dataset(fil0, final_loc='hl', **kwargs) as ds:
        h = (-1 * ds.e.zep().read().np_ops(np.diff, axis=1,
                                           sets_vloc='l')).compute()
        u = ds.u.xsm().read().move_to('h').compute()
        pfu = (-1 * ds.PFu.xsm().read().move_to('h')).compute()
        sigumx = (u * h * pfu).nanmean(axis=0).compute()
    return sigumx


def sigum_loop(fil0, **kwargs):
    with pym6.Dataset(fil0, final_loc='hl', **kwargs) as ds:
        nt = ds.Time.size
        h = (-1 * ds.e.zep().isel(t=slice(0, 1)).read().np_ops(
            np.diff, axis=1, sets_vloc='l')).compute()
        u = ds.u.xsm().isel(t=slice(0, 1)).read().move_to('h').compute()
        pfu = (-1 *
               ds.Pfu.xsm().isel(t=slice(0, 1)).read().move_to('h')).compute()
        sigumx = (u * h * pfu / nt).compute()
        for i in range(1, nt):
            h = (-1 * ds.e.zep().isel(t=slice(i, i + 1)).read().np_ops(
                np.diff, axis=1, sets_vloc='l')).compute()
            u = ds.u.xsm().isel(
                t=slice(i, i + 1)).read().move_to('h').compute()
            pfu = (-1 * ds.Pfu.xsm().isel(
                t=slice(i, i + 1)).read().move_to('h')).compute()
            sigumx.values += (u * h * pfu).compute().values / nt
    return sigumx


def sigvm(fil0, **kwargs):
    with pym6.Dataset(fil0, final_loc='hl', **kwargs) as ds:
        h = (-1 * ds.e.zep().read().np_ops(np.diff, axis=1,
                                           sets_vloc='l')).compute()
        v = ds.v.ysm().read().move_to('h').compute()
        pfv = (-1 * ds.Pfv.ysm().read().move_to('h')).compute()
        sigvmy = (v * h * pfv).nanmean(axis=0).compute()
    return sigvmy


def sigvm_loop(fil0, **kwargs):
    with pym6.Dataset(fil0, final_loc='hl', **kwargs) as ds:
        nt = ds.Time.size
        h = (-1 * ds.e.zep().isel(t=slice(0, 1)).read().np_ops(
            np.diff, axis=1, sets_vloc='l')).compute()
        v = ds.v.ysm().isel(t=slice(0, 1)).read().move_to('h').compute()
        pfv = (-1 *
               ds.Pfv.ysm().isel(t=slice(0, 1)).read().move_to('h')).compute()
        sigvmy = (v * h * pfv / nt).compute()
        for i in range(1, nt):
            h = (-1 * ds.e.zep().isel(t=slice(i, i + 1)).read().np_ops(
                np.diff, axis=1, sets_vloc='l')).compute()
            v = ds.v.ysm().isel(
                t=slice(i, i + 1)).read().move_to('h').compute()
            pfv = (-1 * ds.Pfv.ysm().isel(
                t=slice(i, i + 1)).read().move_to('h')).compute()
            sigvmy.values += (v * h * pfv).compute().values / nt
    return sigvmy


def sigmxuhat(fil1, fil2, **kwargs):
    htol = kwargs.get('htol', 1e-10)
    with pym6.Dataset(
            fil1, final_loc='hl', **kwargs) as ds1, pym6.Dataset(
                fil2, final_loc='hl', **kwargs) as ds2:
        h = ds1.h_Cu.xsm().read().nanmean(axis=0).move_to('h').compute()
        ur = ds2.uh.xsm().read().nanmean(
            axis=0).divide_by('dyCu').move_to('h').compute()
        ur = (ur / h).compute()
        ur.values[h.values < htol] = 0
        hpfu = (-1 * ds1.twa_hpfu.xsm().read().nanmean(axis=0).move_to('h')
                ).compute()
    return (ur * hpfu).compute()


def sigmyvhat(fil1, fil2, **kwargs):
    htol = kwargs.get('htol', 1e-10)
    with pym6.Dataset(
            fil1, final_loc='hl', **kwargs) as ds1, pym6.Dataset(
                fil2, final_loc='hl', **kwargs) as ds2:
        h = ds1.h_Cv.ysm().read().nanmean(axis=0).move_to('h').compute()
        vr = ds2.vh.ysm().read().nanmean(
            axis=0).divide_by('dxCv').move_to('h').compute()
        vr = (vr / h).compute()
        vr.values[h.values < htol] = 0
        hpfv = (-1 * ds1.twa_hpfv.ysm().read().nanmean(axis=0).move_to('h')
                ).compute()
    return (vr * hpfv).compute()


def ekeydivRS(fil1, fil2, **kwargs):
    ydivRS = ty.get_ydivRS(fil1, fil2, **kwargs)
    with pym6.Dataset(fil2, **kwargs) as ds:
        vh = ds.vh.read().nanmean(axis=0).divide_by('dxCv').compute()
    ekeydivRS = ((-vh.compute()) * ydivRS).compute()
    return ekeydivRS


def ekexdivRS(fil1, fil2, **kwargs):
    xdivRS = tx.get_xdivRS(fil1, fil2, **kwargs)
    with pym6.Dataset(fil2, **kwargs) as ds:
        uh = ds.uh.read().nanmean(axis=0).divide_by('dyCu').compute()
    ekexdivRS = ((-uh.compute()) * xdivRS).compute()
    return ekexdivRS
