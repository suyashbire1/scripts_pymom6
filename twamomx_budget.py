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
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
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
        advx = (urx * ur).compute()
        advx.name = 'advx'
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
        advy = (hvm * ury / h).compute()
        advy.name = 'advy'
        return advy


def get_advb(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        urb = ds2.uh.zsm().zep().read().nanmean(
            axis=0).divide_by('dyCu').dbyd(1).move_to('l').compute()
        db = np.diff(ds2.zl)[0] * 9.8 / 1031
        hwm = ds2.wd.xep().zep().read().nanmean(
            axis=0).move_to('l').move_to('u') * db
        advb = (hwm * urb / h).compute()
        advb.name = 'advb'
        return advb


def get_cor(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hfvm = ds1.twa_hfv.read().nanmean(axis=0).compute()
        cor = (hfvm / h).compute()
        cor.name = 'cor'
        return cor


def get_pfum(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        pfum = ds2.PFu.read().nanmean(axis=0).compute()
        pfum = h * pfum / h
        pfum = pfum.compute()
        pfum.name = 'pfu'
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
        initializer['final_loc'] = 'hl'
        e = ds2.e.polish(**initializer).xep().zep().read().nanmean(
            axis=0).move_to('l')**2
        e = e.compute()
        initializer['final_loc'] = 'ul'
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
    xdivep2 = get_advx(fil1, fil2, **initializer)
    xdivep3 = get_xdivep3(fil1, fil2, **initializer)
    xdivep4 = get_xdivep4(fil1, fil2, **initializer)
    xdivep = ((xdivep1 + xdivep2).compute() +
              (xdivep3 + xdivep4).compute()).compute()
    xdivep.name = 'xdivep'
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
    ydivep2 = get_advy(fil1, fil2, **initializer)
    ydivep3 = get_ydivep3(fil1, fil2, **initializer)
    ydivep = (ydivep1 + ydivep2 + ydivep3).compute()
    ydivep.name = 'ydivep'
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
    bdivep2 = get_advb(fil1, fil2, **initializer)
    bdivep3 = get_bdivep3(fil1, fil2, **initializer)
    bdivep4 = get_bdivep4(fil1, fil2, **initializer)
    bdivep = (bdivep1 + bdivep2 + bdivep3 + bdivep4).compute()
    bdivep.name = 'bdivep'
    return bdivep


def get_X1twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdiffum = ds1.twa_hdiffu.read().nanmean(axis=0).compute()
        X1twa = (hdiffum / h).compute()
        X1twa.name = 'X1twa'
        return X1twa


def get_X2twa(fil1, fil2, **initializer):
    initializer['final_loc'] = 'ul'
    h = get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        hdudtviscm = ds1.twa_hdudtvisc.read().nanmean(axis=0).compute()
        X2twa = (hdudtviscm / h).compute()
        X2twa.name = 'X2twa'
        return X2twa


def extract_twamomx_terms(fil1, fil2, **initializer):

    #    returnvars = [
    #        get_advb(fil1, fil2, **initializer),
    #        get_bdivep1(fil1, fil2, **initializer),
    #        get_bdivep3(fil1, fil2, **initializer),
    #        get_bdivep4(fil1, fil2, **initializer),
    #    ]
    #    for var in returnvars:
    #        var = var.compute()
    #        print(np.nanmax(np.fabs(var.array[np.isfinite(var.array)])))

    returnvars = [
        get_advx(fil1, fil2, **initializer),
        get_advy(fil1, fil2, **initializer),
        get_advb(fil1, fil2, **initializer),
        get_cor(fil1, fil2, **initializer),
        get_pfum(fil1, fil2, **initializer),
        get_xdivep(fil1, fil2, **initializer),
        get_ydivep(fil1, fil2, **initializer),
        get_bdivep(fil1, fil2, **initializer),
        get_X1twa(fil1, fil2, **initializer),
        get_X2twa(fil1, fil2, **initializer)
    ]
    return returnvars


def budget_plot(fil1, fil2, **initializer):
    meanax = initializer.get('meanax', 2)
    initializer.pop('meanax')
    z = initializer.get('z', np.linspace(-1200, 0))
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
        b = b.nanmean(axis=meanax).toz(z, e)
        blist[i] = b.to_DataArray()
    blist = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    return blist


if __name__ == '__main__':
    import sys
    fil = sys.argv[1]
    initializer = dict(sys.argv[2])
    extract_twamomx_terms(fil, initializer)

#     ti = [
#         '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)',
#         '(k)', '(l)'
#     ]
#     lab = [
#         r'$-\hat{u}\hat{u}_{\tilde{x}}$', r'$-\hat{v}\hat{u}_{\tilde{y}}$',
#         r'$-\hat{\varpi}\hat{u}_{\tilde{b}}$', r'$f\hat{v}$',
#         r'$-\overline{m_{\tilde{x}}}$',
#         r"""-$\frac{1}{\overline{h}}(\overline{h}\widehat{u ^{\prime \prime} u ^{\prime \prime} })_{\tilde{x}}$""",
#         r"""-$\frac{1}{2\overline{\zeta_{\tilde{b}}}}(\overline{\zeta ^{\prime 2}})_{\tilde{x}}$""",
#         r"""-$\frac{1}{\overline{h}}(\overline{h}\widehat{u ^{\prime \prime} v ^{\prime \prime}})_{\tilde{y}}$""",
#         r"""-$\frac{1}{\overline{h}}(\overline{h}\widehat{u ^{\prime \prime} \varpi ^{\prime \prime}})_{\tilde{b}}$""",
#         r""" -$\frac{1}{2\overline{\zeta_{\tilde{b}}}}(\overline{\zeta ^\prime m_{\tilde{x}}^\prime})_{\tilde{b}}$""",
#         r'$\widehat{X^H}$', r'$\widehat{X^V}$'
#     ]
