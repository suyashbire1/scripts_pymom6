import pymom6.pymom6 as pym6
import numpy as np
import xarray as xr
import pandas as pd
import importlib
importlib.reload(pym6)


def get_KE_budget(fil1, fil2, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil2, **initializer) as ds2:
        pe_to_ke = ds2.PE_to_KE.read().nanmean(axis=0).compute()
        pe_to_ke.name = 'PE to KE'
        pe_to_ke.math = (r'$-(\overline{\sigma u m_{\tilde{x}}}+'
                         r'\overline{\sigma v m_{\tilde{y}}})$')
        ke_adv = ds2.KE_adv.read().nanmean(axis=0).compute()
        ke_adv.name = 'KE_adv'
        ke_adv.math = (r'$-((\overline{\sigma u KE})_{\tilde{x}}+'
                       r'(\overline{\sigma v KE})_{\tilde{y}})$')
        ke_dia = ds2.KE_dia.read().nanmean(axis=0).compute()
        ke_dia.name = 'KE_dia'
        ke_dia.math = r'$-(\overline{\sigma \varpi KE})_{\tilde{b}}$'
        ke_visc = ds2.KE_visc.read().nanmean(axis=0).compute()
        ke_visc.name = 'KE_visc'
        ke_visc.math = '$\overline{\sigma u X^V} + \overline{\sigma v Y^V}$'
        ke_horvisc = ds2.KE_horvisc.read().nanmean(axis=0).compute()
        ke_horvisc.name = 'KE_horvisc'
        ke_horvisc.math = '$\overline{\sigma u X^H} + \overline{\sigma v Y^H}$'

    KE_list = [pe_to_ke, ke_adv, ke_horvisc, ke_visc, ke_dia]
    only = initializer.get('only', range(len(KE_list)))
    return_list = [KE_list[i] for i in only]
    return return_list


def extract_budget(fil1, fil2, fil3=None, **initializer):
    meanax = initializer.get('meanax', 2)
    initializer.pop('meanax')
    z = initializer.get('z', None)
    if 'z' in initializer:
        initializer.pop('z')
    initializer['final_loc'] = 'hi'
    with pym6.Dataset(fil2, **initializer) as ds:
        e = ds.e.zep().read().nanmean(axis=(0, 2)).compute()
    initializer['final_loc'] = 'hl'
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
            swash = swash.ysm().xsm().read().move_to('u').move_to('h').nanmean(
                axis=(0, 2)).compute()
            swash = ((-swash + islaydeepmax) / islaydeepmax * 100).compute()
            if z is not None:
                swash = swash.toz(z, e)
            swash = swash.tokm(3).to_DataArray()
    else:
        swash = None
    blist = get_KE_budget(fil1, fil2, **initializer)
    namelist = []
    for i, b in enumerate(blist):
        namelist.append(b.name)
        b = b.nanmean(axis=meanax)
        if z is not None:
            b = b.toz(z, e)
        blist[i] = b.tokm(3).to_DataArray()
    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
    blist_concat.name = 'Thickness Budget'
    e = e.tokm(3).to_DataArray()
    return_dict = dict(
        blist_concat=blist_concat, blist=blist, e=e, swash=swash)
    return return_dict
