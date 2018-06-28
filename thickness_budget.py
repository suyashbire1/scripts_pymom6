import pymom6.pymom6 as pym6
import numpy as np
import xarray as xr
import pandas as pd
import importlib
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
    blist = get_thickness_budget(fil1, fil2, **initializer)
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
