import pymom6.pymom6 as pym6
import numpy as np
import importlib
import matplotlib.pyplot as plt
importlib.reload(pym6)
import sys


def get_zonal_zeta_adv(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        z = initializer['z']
        nt = ds1.Time.size
        u = ds1.u.xsm().isel(t=slice(0, 1)).read().compute(check_loc=False)
        zx = ds1.e.zep().xsm().xep().isel(
            t=slice(0, 1)).read().dbyd(3).move_to('l').compute(check_loc=False)
        e = ds1.e.final_loc('hi').isel(t=slice(0, 1)).read().compute()
        uzx = ((u * zx).move_to('h').toz(z, e) / nt).compute()
        uzx.name = 'Zonal height adv'
        uzx.math = r'$\hat{u}\bar{\zeta}_{\tilde{x}}$'
        print('Getting uzx')
        for i in range(1, nt):
            u = ds1.u.xsm().isel(t=slice(i, i + 1)).read().compute(
                check_loc=False)
            zx = ds1.e.zep().xsm().xep().isel(
                t=slice(i, i + 1)).read().dbyd(3).move_to('l').compute(
                    check_loc=False)
            e = ds1.e.final_loc('hi').isel(t=slice(i, i + 1)).read().compute()
            uzx.values += (u * zx).move_to('h').toz(z, e).compute().values / nt
    return uzx


def get_merid_zeta_adv(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        z = initializer['z']
        nt = ds1.Time.size
        v = ds1.v.ysm().isel(t=slice(0, 1)).read().compute(check_loc=False)
        zy = ds1.e.zep().ysm().yep().isel(
            t=slice(0, 1)).read().dbyd(2).move_to('l').compute(check_loc=False)
        e = ds1.e.final_loc('hi').isel(t=slice(0, 1)).read().compute()
        vzy = ((v * zy).move_to('h').toz(z, e) / nt).compute()
        vzy.name = 'Merid height adv'
        vzy.math = r'$\hat{v}\bar{\zeta}_{\tilde{y}}$'
        print('Getting vzy')
        for i in range(1, nt):
            v = ds1.v.ysm().isel(t=slice(i, i + 1)).read().compute(
                check_loc=False)
            zy = ds1.e.zep().ysm().yep().isel(
                t=slice(i, i + 1)).read().dbyd(2).move_to('l').compute(
                    check_loc=False)
            e = ds1.e.final_loc('hi').isel(t=slice(i, i + 1)).read().compute()
            vzy.values += (
                (v * zy).move_to('h').toz(z, e) / nt).compute().values
    return vzy


def get_w(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    uzx = get_zonal_zeta_adv(fil1, **initializer)
    vzy = get_merid_zeta_adv(fil1, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        z = initializer['z']
        nt = ds1.Time.size
        e = ds1.e.final_loc('hi').isel(t=slice(0, 1)).read().compute()
        wd = (ds1.wd.zep().isel(t=slice(0, 1)).read().move_to('l') / nt).toz(
            z, e).compute()
        wd = (-wd).compute()
        for i in range(1, nt):
            e = ds1.e.final_loc('hi').isel(t=slice(i, i + 1)).read().compute()
            wd.values -= (
                ds1.wd.zep().isel(t=slice(i, i + 1)).read().move_to('l').toz(
                    z, e) / nt).compute().values
    w = (wd + uzx + vzy).compute()
    w.name = 'Mean vert vel'
    w.math = r'$\bar{w}^z$'
    return w


def get_velocities(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        z = initializer['z']
        nt = ds1.Time.size
        e = ds1.e.final_loc('hi').isel(t=slice(0, 1)).read().compute()
        v = (ds1.v.ysm().isel(t=slice(0, 1)).read().move_to('h').toz(z, e) /
             nt).compute(check_loc=False)
        v.name = 'Mean merid vel'
        v.math = r'$\bar{v}^z$'

        u = (ds1.u.xsm().isel(t=slice(0, 1)).read().move_to('h').toz(z, e) /
             nt).compute(check_loc=False)
        u.name = 'Mean zonal vel'
        u.math = r'$\bar{u}^z$'
        print('Getting u, v...')
        for i in range(1, nt):
            e = ds1.e.final_loc('hi').isel(t=slice(i, i + 1)).read().compute()
            v.values += (ds1.v.ysm().isel(
                t=slice(i, i + 1)).read().move_to('h').toz(z, e) / nt).compute(
                    check_loc=False).values
            u.values += (ds1.u.xsm().isel(
                t=slice(i, i + 1)).read().move_to('h').toz(z, e) / nt).compute(
                    check_loc=False).values

        print('Getting w...')
        w = get_w(fil1, **initializer)

    ret_dict = dict(v=v, u=u, w=w)
    meanax = initializer.get('meanax', None)
    tokm = initializer.get('tokm', 3)
    for key, value in ret_dict.items():
        if meanax is not None:
            value = value.nanmean(axis=meanax)
        if tokm is not None:
            value = value.tokm(tokm)
        ret_dict[key] = value.to_DataArray(check_loc=False)
    return ret_dict


def print_details(dic):
    for key, value in dic.items():
        print(value.name, value.shape)
        for key1, value1 in value.dimensions.items():
            print(key1, value1.size)


def plot_one_panel(key, ax, alphabet, returned_dict, fil1, fil2,
                   **initializer):
    vmin = initializer.get('vmin', None)
    vmax = initializer.get('vmax', None)
    eskip = initializer.get('eskip', 1)
    textx = initializer.get('textx', 0.03)
    texty = initializer.get('texty', 0.1)
    fontsize = initializer.get('fontsize', 15)
    interpolation = initializer.get('interpolation', 'none')
    name = initializer.get('name', True)
    z = initializer.get('z', None)
    im = returned_dict[key].squeeze().plot.imshow(
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        cmap='RdBu_r',
        add_colorbar=False,
        add_labels=False)
    ax.set_xlabel(list(returned_dict[key].squeeze(drop=True).coords)[1])
    if ax == ax.get_figure().axes[0]:
        ax.set_ylabel(list(returned_dict[key].squeeze(drop=True).coords)[0])
    cb = plt.colorbar(im, ax=ax)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    if z is not None:
        meanax = initializer.get('meanax', None)
        tokm = initializer.get('tokm', 3)
        with pym6.Dataset(fil2, **initializer) as ds1:
            e = ds1.e.zep().read().compute()
        if meanax is not None:
            e = e.nanmean(axis=(0, meanax))
        if tokm is not None:
            e = e.tokm(3)
        e = e.to_DataArray()
        ax.plot(
            e.squeeze(drop=True).coords[e.squeeze(drop=True).dims[1]],
            e.values.squeeze()[::eskip, :].T,
            'k',
            lw=1)
    if name:
        ax.set_title('(' + alphabet + ') ' + returned_dict[key].name)
    else:
        ax.set_title('(' + alphabet + ') ')
    ax.text(
        textx,
        texty,
        returned_dict[key].math,
        transform=ax.transAxes,
        fontsize=fontsize,
        bbox=dict(facecolor='w', edgecolor='w', alpha=0.8))
    return im


def plot_velocities(dict_or_func, fil1, fil2, **initializer):
    if callable(dict_or_func):
        returned_dict = dict_or_func(fil1, **initializer)
    else:
        returned_dict = dict_or_func

    figsize = initializer.get('figsize', (5, 5))
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)

    plot_one_panel('u', ax[0], 'a', returned_dict, fil1, fil2, **initializer)
    plot_one_panel('v', ax[1], 'b', returned_dict, fil1, fil2, **initializer)
    plot_one_panel('w', ax[2], 'c', returned_dict, fil1, fil2, **initializer)
    return fig


def get_v_accurate(fil1, **initializer):
    with pym6.Dataset(fil1, **initializer) as ds1:
        z = initializer['ev']
        dz = initializer['dzv']
        nt = ds1.Time.size
        e = ds1.e.final_loc('vi').isel(
            t=slice(0, 1)).yep().read().move_to('v').compute()
        v = ds1.v.isel(t=slice(0, 1)).read().compute()
        h = (-ds1.e.final_loc('vl').isel(
            t=slice(0, 1)).yep().zep().read().move_to('v').np_ops(
                np.diff, axis=1, sets_vloc='l')).compute()
        transport = (v * h).compute()
        temp = transport.values.copy()
        shape = list(transport.shape)
        shape[1] += 1
        transport.values = np.zeros(shape)
        transport.values[:, 1:] = np.cumsum(temp, axis=1)
        v_new = (transport.toz(z, e, linear=True).np_ops(np.diff, axis=1) / dz
                 / nt).compute()
        v_new.name = 'Mean merid vel'
        v_new.math = r'$\bar{v}^z$'

        print('Getting v...')
        for i in range(1, nt):
            e = ds1.e.final_loc('vi').isel(
                t=slice(i, i + 1)).yep().read().move_to('v').compute()
            v = ds1.v.isel(t=slice(i, i + 1)).read().compute()
            h = (-ds1.e.final_loc('vl').isel(
                t=slice(i, i + 1)).yep().zep().read().move_to('v').np_ops(
                    np.diff, axis=1, sets_vloc='l')).compute()
            transport = (v * h).compute()
            temp = transport.values.copy()
            shape = list(transport.shape)
            shape[1] += 1
            transport.values = np.zeros(shape)
            transport.values[:, 1:] = np.cumsum(temp, axis=1)
            v_new1 = (transport.toz(z, e, linear=True).np_ops(np.diff, axis=1)
                      / dz / nt).compute()
            v_new.values += v_new1.values
            sys.stdout.write(f"\r{i/nt*100:.2f}")
            sys.stdout.flush()
        return v_new


def get_u_accurate(fil1, **initializer):
    with pym6.Dataset(fil1, **initializer) as ds1:
        z = initializer['eu']
        dz = initializer['dzu']
        nt = ds1.Time.size

        e = ds1.e.final_loc('ui').isel(
            t=slice(0, 1)).xep().read().move_to('u').compute()
        u = ds1.u.isel(t=slice(0, 1)).read().compute()
        h = (-ds1.e.final_loc('ul').isel(
            t=slice(0, 1)).xep().zep().read().move_to('u').np_ops(
                np.diff, axis=1, sets_vloc='l')).compute()
        transport = (u * h).compute()
        temp = transport.values.copy()
        shape = list(transport.shape)
        shape[1] += 1
        transport.values = np.zeros(shape)
        transport.values[:, 1:] = np.cumsum(temp, axis=1)
        u_new = (transport.toz(z, e, linear=True).np_ops(np.diff, axis=1) / dz
                 / nt).compute()
        u_new.name = 'Mean zonal vel'
        u_new.math = r'$\bar{u}^z$'
        print('Getting u...')
        for i in range(1, nt):

            e = ds1.e.final_loc('ui').isel(
                t=slice(i, i + 1)).xep().read().move_to('u').compute()
            u = ds1.u.isel(t=slice(i, i + 1)).read().compute()
            h = (-ds1.e.final_loc('ul').isel(
                t=slice(i, i + 1)).xep().zep().read().move_to('u').np_ops(
                    np.diff, axis=1, sets_vloc='l')).compute()
            transport = (u * h).compute()
            temp = transport.values.copy()
            shape = list(transport.shape)
            shape[1] += 1
            transport.values = np.zeros(shape)
            transport.values[:, 1:] = np.cumsum(temp, axis=1)
            u_new1 = (transport.toz(z, e, linear=True).np_ops(np.diff, axis=1)
                      / dz / nt).compute()
            u_new.values += u_new1.values
            sys.stdout.write(f"\r{i/nt*100:.2f}")
            sys.stdout.flush()
        return u_new
