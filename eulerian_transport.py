import pymom6.pymom6 as pym6
import importlib
import matplotlib.pyplot as plt
import numpy as np
importlib.reload(pym6)


def get_transport(fil1, **initializer):
    #initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        z = initializer['z']
        nt = ds1.Time.size
        e = ds1.e.final_loc('vi').yep().isel(
            t=slice(0, 1)).read().move_to('v').compute()
        vh = ds1.v.isel(t=slice(0, 1)).read().multiply_by('dxCv').compute()
        h = (-ds1.e.final_loc('vl').yep().zep().isel(
            t=slice(0, 1)).read().move_to('v').np_ops(
                np.diff, axis=1, sets_vloc='l')).compute()
        vh = (vh * h).compute()
        vh.values = np.cumsum(vh.values, axis=1)
        vh.values = np.concatenate((np.zeros(
            (vh.shape[0], 1, vh.shape[2], vh.shape[3])), vh.values),
                                   axis=1)
        vh = (vh.toz(z, e, linear=True) / nt).compute()

        vh.name = 'Mean merid transport'
        vh.math = r'$\bar{vh}^z$'

        e = ds1.e.final_loc('ui').xep().isel(
            t=slice(0, 1)).read().move_to('u').compute()
        uh = ds1.u.isel(t=slice(0, 1)).read().multiply_by('dyCu').compute()
        h = (-ds1.e.final_loc('ul').xep().zep().isel(
            t=slice(0, 1)).read().move_to('u').np_ops(
                np.diff, axis=1, sets_vloc='l')).compute()
        uh = (uh * h).compute()
        uh.values = np.cumsum(uh.values, axis=1)
        uh.values = np.concatenate((np.zeros(
            (uh.shape[0], 1, uh.shape[2], uh.shape[3])), uh.values),
                                   axis=1)
        uh = (uh.toz(z, e, linear=True) / nt).compute()
        uh.name = 'Mean zonal vel'
        uh.math = r'$\bar{u}^z$'
        print('Getting u, v...')
        for i in range(1, nt):
            e = ds1.e.final_loc('vi').yep().isel(
                t=slice(i, i + 1)).read().move_to('v').compute()
            vh1 = ds1.v.isel(
                t=slice(i, i + 1)).read().multiply_by('dxCv').compute()
            h = (-ds1.e.final_loc('vl').yep().zep().isel(
                t=slice(i, i + 1)).read().move_to('v').np_ops(
                    np.diff, axis=1, sets_vloc='l')).compute()
            vh1 = (vh1 * h).compute()
            vh1.values = np.cumsum(vh1.values, axis=1)
            vh1.values = np.concatenate((np.zeros(
                (vh1.shape[0], 1, vh1.shape[2], vh1.shape[3])), vh1.values),
                                        axis=1)
            vh1 = (vh1.toz(z, e, linear=True) / nt).compute().values
            vh.values += vh1

            e = ds1.e.final_loc('ui').xep().isel(
                t=slice(i, i + 1)).read().move_to('u').compute()
            uh1 = ds1.u.isel(
                t=slice(i, i + 1)).read().multiply_by('dyCu').compute()
            h = (-ds1.e.final_loc('ul').xep().zep().isel(
                t=slice(i, i + 1)).read().move_to('u').np_ops(
                    np.diff, axis=1, sets_vloc='l')).compute()
            uh1 = (uh1 * h).compute()
            uh1.values = np.cumsum(uh1.values, axis=1)
            uh1.values = np.concatenate((np.zeros(
                (uh1.shape[0], 1, uh1.shape[2], uh1.shape[3])), uh1.values),
                                        axis=1)
            uh1 = (uh1.toz(z, e, linear=True) / nt).compute().values
            uh.values += uh1
            print(i, nt)

    ret_dict = dict(vh=vh, uh=uh)
    for key, value in ret_dict.items():
        ret_dict[key] = value.to_DataArray()
    return ret_dict


def get_transport_at_zl(fil1, **initializer):
    #initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        zu = initializer['zu']
        zv = initializer['zv']
        nt = ds1.Time.size
        e = ds1.e.final_loc('vi').yep().isel(
            t=slice(0, 1)).read().move_to('v').compute()
        vh = ds1.v.isel(t=slice(0, 1)).read().multiply_by('dxCv').compute()
        h = (-ds1.e.final_loc('vl').yep().zep().isel(
            t=slice(0, 1)).read().move_to('v').np_ops(
                np.diff, axis=1, sets_vloc='l')).compute()
        vh = (vh * h).compute()
        vh.values = np.cumsum(vh.values, axis=1)
        vh.values = np.concatenate((np.zeros(
            (vh.shape[0], 1, vh.shape[2], vh.shape[3])), vh.values),
                                   axis=1)
        vh = (vh.toz(zv, e, linear=True) / nt).compute()

        vh.name = 'Mean merid transport'
        vh.math = r'$\bar{vh}^z$'

        e = ds1.e.final_loc('ui').xep().isel(
            t=slice(0, 1)).read().move_to('u').compute()
        uh = ds1.u.isel(t=slice(0, 1)).read().multiply_by('dyCu').compute()
        h = (-ds1.e.final_loc('ul').xep().zep().isel(
            t=slice(0, 1)).read().move_to('u').np_ops(
                np.diff, axis=1, sets_vloc='l')).compute()
        uh = (uh * h).compute()
        uh.values = np.cumsum(uh.values, axis=1)
        uh.values = np.concatenate((np.zeros(
            (uh.shape[0], 1, uh.shape[2], uh.shape[3])), uh.values),
                                   axis=1)
        uh = (uh.toz(zu, e, linear=True) / nt).compute()
        uh.name = 'Mean zonal vel'
        uh.math = r'$\bar{u}^z$'
        print('Getting u, v...')
        for i in range(1, nt):
            e = ds1.e.final_loc('vi').yep().isel(
                t=slice(i, i + 1)).read().move_to('v').compute()
            vh1 = ds1.v.isel(
                t=slice(i, i + 1)).read().multiply_by('dxCv').compute()
            h = (-ds1.e.final_loc('vl').yep().zep().isel(
                t=slice(i, i + 1)).read().move_to('v').np_ops(
                    np.diff, axis=1, sets_vloc='l')).compute()
            vh1 = (vh1 * h).compute()
            vh1.values = np.cumsum(vh1.values, axis=1)
            vh1.values = np.concatenate((np.zeros(
                (vh1.shape[0], 1, vh1.shape[2], vh1.shape[3])), vh1.values),
                                        axis=1)
            vh1 = (vh1.toz(zv, e, linear=True) / nt).compute().values
            vh.values += vh1

            e = ds1.e.final_loc('ui').xep().isel(
                t=slice(i, i + 1)).read().move_to('u').compute()
            uh1 = ds1.u.isel(
                t=slice(i, i + 1)).read().multiply_by('dyCu').compute()
            h = (-ds1.e.final_loc('ul').xep().zep().isel(
                t=slice(i, i + 1)).read().move_to('u').np_ops(
                    np.diff, axis=1, sets_vloc='l')).compute()
            uh1 = (uh1 * h).compute()
            uh1.values = np.cumsum(uh1.values, axis=1)
            uh1.values = np.concatenate((np.zeros(
                (uh1.shape[0], 1, uh1.shape[2], uh1.shape[3])), uh1.values),
                                        axis=1)
            uh1 = (uh1.toz(zu, e, linear=True) / nt).compute().values
            uh.values += uh1
            print(i, nt)

    ret_dict = dict(vh=vh, uh=uh)
    return ret_dict
