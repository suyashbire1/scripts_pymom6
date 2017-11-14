import pymom6.pymom6 as pym6
import numpy as np
import importlib
import twamomx_budget as tx
import twamomy_budget as ty
import xarray as xr
import pandas as pd
import string
importlib.reload(pym6)
importlib.reload(tx)
importlib.reload(ty)


def get_domain(fil1, fil2, **initializer):
    with pym6.Dataset(fil2, **initializer) as ds:
        indices = ds.uh.indices
        sx, ex, stridex = indices['xq']
        sy, ey, stridey = indices['yq']

    initializer['by_index'] = True
    initializer['sx'] = sx
    initializer['ex'] = ex
    initializer['sy'] = sy
    initializer['ey'] = ey
    initializer['stridex'] = stridex
    initializer['stridey'] = stridey
    return initializer


def extract_twapv_terms(fil1, fil2, **initializer):

    initializer = get_domain(fil1, fil2, **initializer)
    momy = ty.extract_twamomy_terms(
        fil1, fil2, type='withPVflux', **initializer)
    initializer['ey'] += 1
    momx = tx.extract_twamomx_terms(
        fil1, fil2, type='withPVflux', **initializer)
    initializer['ey'] -= 1

    PV = get_PV(fil1, fil2, **initializer)
    uhx, vhy, dwd, h = get_thickness_budget(fil1, fil2, **initializer)

    pv_budget = []

    advx = (
        (momy[0].xep().implement_BC_if_necessary().dbyd(3).compute(
            check_loc=False) +
         (uhx * PV).compute(check_loc=False)) / h).compute(check_loc=False)
    pv_budget.append(advx)
    advx.name = 'Zonal advection'
    advx.math = r'-$\hat{u}\Pi^{\sharp}_{\tilde{x}}$'

    advy = (
        (momx[0].dbyd(2).compute(check_loc=False) +
         (vhy * PV).compute(check_loc=False)) / h).compute(check_loc=False)
    pv_budget.append(advy)
    advy.name = 'Meridional advection'
    advy.math = r'-$\hat{v}\Pi^{\sharp}_{\tilde{y}}$'

    lab = [
        'Diabatic effect', 'Div grad Bernoulli func', 'Zonal Reynolds stress',
        'Merid Reynolds stress', 'Eddy form drag', 'Diffusion',
        'Vertical friction', 'Vertical advection'
    ]
    math = [
        (r"""$\frac{(\hat{\varpi} \hat{u}_{\tilde{b}})_{\tilde{y}} -"""
         r"""(\hat{\varpi} \hat{v}_{\tilde{b}})_{\tilde{x}}}{\bar{\sigma}}$"""
         ), r"""$-B_{\tilde{x}\tilde{y}} + B_{\tilde{x}\tilde{y}}$""",
        (r"""-$\frac{1}{\bar{\sigma}}(\frac{1}{\bar{\sigma}}"""
         r"""(\bar{\sigma}\widehat{u ^{\prime \prime} """
         r"""v ^{\prime \prime}})_{\tilde{x}})_{\tilde{x}}$"""),
        (r"""-$\frac{1}{\bar{\sigma}}(\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
         r"""\widehat{v ^{\prime \prime} v ^{\prime \prime} })_{\tilde{y}}$"""
         ), (r""" -$\frac{1}{\bar{\sigma}}(\frac{1}{2\bar{\sigma}}"""
             r"""(\overline{\zeta ^\prime m_{\tilde{y}}^\prime})"""
             r"""_{\tilde{b}})_{\tilde{x}}$"""),
        (r"""$\frac{\hat{Y}^H_{\tilde{x}}"""
         r"""- \hat{\hat{X}^H_{\tilde{y}}}}{{\bar{\sigma}}}$"""),
        (r"""$\frac{\hat{Y}^V_{\tilde{x}}"""
         r"""- \hat{\hat{X}^V_{\tilde{y}}}}{{\bar{\sigma}}}$"""),
        (r"""$\frac{\bar{\sigma}\hat{\varpi}\Pi^{\sharp}}{\bar{\sigma}}"""
         r"""- \hat{\varpi}\Pi^{\sharp}_{\tilde{b}}$""")
    ]
    for i, (y, x) in enumerate(zip(momy[1:], momx[1:])):
        y = y.xep().implement_BC_if_necessary().dbyd(3).compute(
            check_loc=False)
        x = x.dbyd(2).compute(check_loc=False)
        ymx = ((y - x) / h).compute(check_loc=False)
        ymx.name = lab[i]
        ymx.math = math[i]
        pv_budget.append(ymx)

    advb = ((dwd * PV) / h).compute(check_loc=False)
    advb.name = lab[-1]
    advb.math = math[-1]
    pv_budget.append(advb)

    return pv_budget


def get_PV(fil1, fil2, **initializer):
    htol = initializer.get('htol', 1e-3)
    initializer['ey'] += 1
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_fory = ds1.h_Cu.read().nanmean(axis=0).compute()
        a = h_fory.values
        a[a < htol] = np.nan
        h_fory.values = a
        ury = ds2.uh.read().nanmean(axis=0).divide_by('dyCu') / h_fory
        ury = ury.dbyd(2)
        f_slice = ury.get_slice_2D()._slice_2D
        f = initializer['geometry'].f[f_slice]
        ury = ury.compute(check_loc=False)
    initializer['ey'] -= 1
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_forx = ds1.h_Cv.xep().read().nanmean(axis=0).compute(check_loc=False)
        a = h_forx.values
        a[a < htol] = np.nan
        h_forx.values = a
        vrx = ds2.vh.xep().read().nanmean(axis=0).divide_by('dxCv') / h_forx
        vrx = vrx.dbyd(3).compute(check_loc=False)
        h = h_forx.move_to('q').compute(check_loc=False)
    PV = ((vrx - ury + f) / h).compute(check_loc=False)
    PV.name = r'PV$^\sharp$'
    return PV


def get_thickness_budget(fil1, fil2, **initializer):
    initializer['ey'] += 1
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        dwd = ds2.wd.xep().zep().read().nanmean(axis=0).np_ops(
            np.diff, axis=1,
            sets_vloc='l').move_to('u').move_to('q').compute(check_loc=False)
        uhx = ds2.uh.xsm().xep().read().nanmean(axis=0).divide_by('dyCu').dbyd(
            3).move_to('u').move_to('q').compute(check_loc=False)
        h = ds1.h_Cu.read().nanmean(axis=0).move_to('q').compute(
            check_loc=False)
        a = h.values
        a[a < htol] = np.nan
        h.values = a
    initializer['ey'] -= 1
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        vhy = ds2.vh.ysm().yep().xep().read().nanmean(axis=0).divide_by(
            'dxCv').dbyd(2).move_to('u').move_to('q').compute(check_loc=False)
    return uhx, vhy, dwd, h


def get_beta_term(fil1, fil2, **initializer):
    initializer = get_domain(fil1, fil2, **initializer)
    initializer['ey'] += 1
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        ury = ds2.uh.read().nanmean(axis=0).divide_by('dyCu')
        f_slice = ury.get_slice_2D()._slice_2D
        f = initializer['geometry'].f[f_slice]
        ury = ury.dbyd(2)
        f_slice = ury.get_slice_2D()._slice_2D
        beta = np.diff(f, axis=0) / initializer['geometry'].dyCu[f_slice]

    initializer['ey'] -= 1
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h = ds1.h_Cv.xep().read().nanmean(axis=0).compute(check_loc=False)
        a = h.values
        htol = initializer.get('htol', 1e-3)
        a[a < htol] = np.nan
        h.values = a
        vr = ds2.vh.xep().read().nanmean(axis=0).divide_by('dxCv') / h
        vr = vr.move_to('q').compute(check_loc=False)
        h = h.move_to('q').compute(check_loc=False)
        beta_term = (vr * beta / h).compute(check_loc=False)
    return beta_term


def extract_budget(fil1, fil2, **initializer):
    blist = extract_twapv_terms(fil1, fil2, **initializer)
    meanax = initializer.get('meanax', 2)
    initializer.pop('meanax')
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
        initializer['final_loc'] = 'qi'
        with pym6.Dataset(fil2, **initializer) as ds:
            e = ds.e.xep().zep().yep().read().move_to('u').move_to(
                'q').nanmean(axis=(0, 2)).compute()
    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.nanmean(axis=meanax)
        if z is not None:
            b = b.toz(z, e).compute(check_loc=False)
        blist[i] = b.to_DataArray(check_loc=False)
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'TWA PV budget'
    return blist_concat, blist


def plot_budget(fil1, fil2, **initializer):
    blist_concat, blist = extract_budget(fil1, fil2, **initializer)
    fg = blist_concat.plot.imshow(
        'xq',
        'z',
        size=2,
        aspect=(1 + np.sqrt(5)) / 2,
        yincrease=True,
        vmin=-5e-12,
        vmax=5e-12,
        cmap='RdBu_r',
        col='Term',
        col_wrap=5,
        robust=True)
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
    extract_twapv_terms(fil, initializer)
