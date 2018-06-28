import pymom6.pymom6 as pym6
import numpy as np
import importlib
import twamomx_budget as tx
import twamomy_budget as ty
import xarray as xr
import pandas as pd
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

    only = initializer.get('only', None)
    if 'only' in initializer:
        initializer.pop('only')
    initializer = get_domain(fil1, fil2, **initializer)
    momy = ty.extract_twamomy_terms(fil1, fil2, type='forPV', **initializer)
    initializer['ey'] += 1
    momx = tx.extract_twamomx_terms(fil1, fil2, type='forPV', **initializer)
    initializer['ey'] -= 1

    PV = get_PV(fil1, fil2, **initializer)
    uhx, vhy, dwd, h = get_thickness_budget(fil1, fil2, **initializer)

    pv_budget = []

    advx = (
        (momy[0].xep().implement_BC_if_necessary().multiply_by('dyCv').dbyd(
            3, weights='area').compute(check_loc=False) +
         (uhx * PV).compute(check_loc=False)) / h).compute(check_loc=False)
    advx.name = 'Zonal advection'
    advx.math = r'-$\hat{u}\Pi^{\sharp}_{\tilde{x}}$'
    pv_budget.append(advx)

    advy = (
        (-momx[0].multiply_by('dxCu').dbyd(
            2, weights='area').compute(check_loc=False) +
         (vhy * PV).compute(check_loc=False)) / h).compute(check_loc=False)
    advy.name = 'Meridional advection'
    advy.math = r'-$\hat{v}\Pi^{\sharp}_{\tilde{y}}$'
    pv_budget.append(advy)

    lab = [
        'Diabatic effect', 'Div grad Bernoulli func', 'Zonal Reynolds stress',
        'Merid Reynolds stress', 'Vert Reynolds stress', 'Horizontal friction',
        'Vertical friction', 'EPE', 'Merid eddy form drag',
        'Zonal eddy form drag', 'Vertical advection'
    ]
    math = [
        (r"""$\frac{(\hat{\varpi} \hat{u}_{\tilde{b}})_{\tilde{y}} -"""
         r"""(\hat{\varpi} \hat{v}_{\tilde{b}})_{\tilde{x}}}{\bar{\sigma}}$"""
         ), r"""$-B_{\tilde{x}\tilde{y}} + B_{\tilde{x}\tilde{y}}$""",
        (r"""-$\frac{1}{\bar{\sigma}}(\frac{1}{\bar{\sigma}}"""
         r"""(\bar{\sigma}\widehat{u ^{\prime \prime} """
         r"""v ^{\prime \prime}})_{\tilde{x}})_{\tilde{x}}$"""),
        (r"""-$\frac{1}{\bar{\sigma}}(\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
         r"""\widehat{v ^{\prime \prime} v ^{\prime \prime} })"""
         r"""_{\tilde{y}})_{\tilde{x}}$"""),
        (r"""-$\frac{1}{\bar{\sigma}}((\frac{1}{\bar{\sigma}}(\bar{\sigma}"""
         r"""\widehat{\varpi ^{\prime \prime} v ^{\prime \prime} }"""
         r"""_{\tilde{b}}))_{tilde{x}} -"""
         """(\frac{1}{\bar{\sigma}}"""
         r"""(\widehat{\varpi ^{\prime \prime} u ^{\prime \prime}}"""
         r"""_{\tilde{b}}))_tilde{y})$"""),
        (r"""$\frac{\hat{Y}^H_{\tilde{x}}}{{\bar{\sigma}}}$"""),
        (r"""$\frac{\hat{Y}^V_{\tilde{x}}"""
         r"""- \hat{\hat{X}^V_{\tilde{y}}}}{{\bar{\sigma}}}$"""), ("""EPE"""),
        (r""" -$\frac{1}{\bar{\sigma}}(\frac{1}{\bar{\sigma}}"""
         r"""(\overline{\zeta ^\prime m_{\tilde{y}}^\prime})"""
         r"""_{\tilde{b}})_{\tilde{x}}$"""),
        (r""" -$\frac{1}{\bar{\sigma}}(\frac{1}{\bar{\sigma}}"""
         r"""(\overline{\zeta ^\prime m_{\tilde{x}}^\prime})"""
         r"""_{\tilde{b}})_{\tilde{y}}$"""),
        (r"""$\frac{(\bar{\sigma}\hat{\varpi}\Pi^{\sharp})_{\tilde{b}}}"""
         r"""{\bar{\sigma}} - \hat{\varpi}\Pi^{\sharp}_{\tilde{b}}$""")
    ]
    for i, (y, x) in enumerate(zip(momy[1:9], momx[1:9])):
        y = y.xep().implement_BC_if_necessary().multiply_by('dyCv').dbyd(
            3, weights='area').compute(check_loc=False)
        x = x.multiply_by('dxCu').dbyd(
            2, weights='area').compute(check_loc=False)
        if i == 2:
            ymx = (y / h).compute(check_loc=False)
        else:
            ymx = ((y - x) / h).compute(check_loc=False)
        ymx.name = lab[i]
        ymx.math = math[i]
        pv_budget.append(ymx)

    formdragy = momy[9].xep().implement_BC_if_necessary().multiply_by(
        'dyCv').dbyd(
            3, weights='area').compute(check_loc=False)
    formdragy = (formdragy / h).compute(check_loc=False)
    formdragy.name = lab[8]
    formdragy.math = (r""" -$\frac{1}{\bar{\sigma}}(\frac{1}{\bar{\sigma}}"""
                      r"""(\overline{\zeta ^\prime m_{\tilde{y}}^\prime})"""
                      r"""_{\tilde{b}})_{\tilde{x}}$""")
    formdragx = (-momx[9]).compute(check_loc=False).multiply_by('dxCu').dbyd(
        2, weights='area').compute(check_loc=False)
    formdragx = (formdragx / h).compute(check_loc=False)
    formdragx.name = lab[9]
    formdragx.math = (r""" -$\frac{1}{\bar{\sigma}}(\frac{1}{\bar{\sigma}}"""
                      r"""(\overline{\zeta ^\prime m_{\tilde{x}}^\prime})"""
                      r"""_{\tilde{b}})_{\tilde{y}}$""")
    pv_budget.append(formdragy)
    pv_budget.append(formdragx)

    advb = ((dwd * PV) / h).compute(check_loc=False)
    advb.name = lab[-1]
    advb.math = math[-1]
    pv_budget.append(advb)

    with pym6.Dataset(fil1, **initializer) as ds1:
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
    pv_budget_new = []
    for term in pv_budget:
        pv_budget_new.append(term * db)

    if only is None:
        return_pv_budget = pv_budget_new
    else:
        return_pv_budget = [pv_budget_new[i] for i in only]
    return return_pv_budget, PV


def get_PV(fil1, fil2, **initializer):
    htol = initializer.get('htol', 1e-3)
    initializer['ey'] += 1
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h_fory = ds1.h_Cu.read().nanmean(axis=0).compute()
        a = h_fory.values
        a[a < htol] = np.nan
        h_fory.values = a
        ury = ds2.uh.read().nanmean(axis=0) / h_fory
        ury = ury.dbyd(2, weights='area')
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
        vrx = ds2.vh.xep().read().nanmean(axis=0) / h_forx
        vrx = vrx.dbyd(3, weights='area').compute(check_loc=False)
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
        uhx = ds2.uh.xsm().xep().read().nanmean(axis=0).dbyd(
            3,
            weights='area').move_to('u').move_to('q').compute(check_loc=False)
        h = ds1.h_Cu.read().nanmean(axis=0).move_to('q').compute(
            check_loc=False)
        a = h.values
        a[a < htol] = np.nan
        h.values = a
    initializer['ey'] -= 1
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        vhy = ds2.vh.ysm().yep().xep().read().nanmean(axis=0).dbyd(
            2,
            weights='area').move_to('u').move_to('q').compute(check_loc=False)
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


def extract_budget(fil1, fil2, fil3=None, **initializer):
    blist, PV = extract_twapv_terms(fil1, fil2, **initializer)
    meanax = initializer.get('meanax', 2)
    initializer.pop('meanax')
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'qi'
    with pym6.Dataset(fil2, **initializer) as ds:
        e = ds.e.xep().zep().yep().read().move_to('u').move_to('q').nanmean(
            axis=(0, 2)).compute()
    initializer['final_loc'] = 'ql'
    if fil3 is not None:
        with pym6.Dataset(fil3) as ds:
            islaydeepmax = ds.islayerdeep.read().compute(check_loc=False).array
            islaydeepmax = islaydeepmax[0, 0, 0, 0]
        with pym6.Dataset(fil3, fillvalue=np.nan, **initializer) as ds:
            swash = ds.islayerdeep
            swash = swash.read().nanmean(axis=(0, 2)).compute()
            swash = ((-swash + islaydeepmax) / islaydeepmax * 100).compute()
            if z is not None:
                swash = swash.toz(z, e)
            swash = swash.tokm(3).to_DataArray()
    else:
        swash = None
    PV = PV.nanmean(axis=meanax)
    if z is not None:
        PV = PV.toz(z, e)
    PV = PV.tokm(3).to_DataArray(check_loc=False)
    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.nanmean(axis=meanax)
        if z is not None:
            b = b.toz(z, e).compute(check_loc=False)
        blist[i] = b.tokm(3).to_DataArray(check_loc=False)
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'TWA PV budget'
    e = e.tokm(3).to_DataArray()
    withPV = initializer.get('withPV', True)
    if withPV:
        return_dict = dict(
            blist_concat=blist_concat, blist=blist, e=e, PV=PV, swash=swash)
    else:
        return_dict = dict(
            blist_concat=blist_concat, blist=blist, e=e, swash=swash)
    return return_dict
