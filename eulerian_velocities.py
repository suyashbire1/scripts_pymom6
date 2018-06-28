import pymom6.pymom6 as pym6
import importlib
import matplotlib.pyplot as plt
importlib.reload(pym6)


def get_zonal_zeta_adv(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        u = ds1.u.xsm().read().compute(check_loc=False)
        zx = ds1.e.zep().xsm().xep().read().dbyd(3).move_to('l').compute(
            check_loc=False)
        uzx = (u * zx).move_to('h').compute()
        uzx.name = 'Zonal height adv'
        uzx.math = r'$\hat{u}\bar{\zeta}_{\tilde{x}}$'
    return uzx


def get_merid_zeta_adv(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        v = ds1.v.ysm().read().compute(check_loc=False)
        zy = ds1.e.zep().ysm().yep().read().dbyd(2).move_to('l').compute(
            check_loc=False)
        vzy = (v * zy).move_to('h').compute()
        vzy.name = 'Merid height adv'
        vzy.math = r'$\hat{v}\bar{\zeta}_{\tilde{y}}$'
    return vzy


def get_w(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    uzx = get_zonal_zeta_advds(fil1, **initializer)
    vzy = get_merid_zeta_advds(fil1, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1:
        wd = ds1.wd.zep().read().move_to('l').compute()
    w = (wd + uzx + vzy).compute()
    w.name = 'Mean vert vel'
    w.math = r'$\bar{w}^z$'
    return w


def get_zonal_zeta_advds(ds):
    u = ds.u.xsm().read().compute(check_loc=False)
    zx = ds.e.zep().xsm().xep().read().dbyd(3).move_to('l').compute(
        check_loc=False)
    uzx = (u * zx).move_to('h').compute()
    uzx.name = 'Zonal height adv'
    uzx.math = r'$\hat{u}\bar{\zeta}_{\tilde{x}}$'
    return uzx


def get_merid_zeta_advds(ds):
    v = ds.v.ysm().read().compute(check_loc=False)
    zy = ds.e.zep().ysm().yep().read().dbyd(2).move_to('l').compute(
        check_loc=False)
    vzy = (v * zy).move_to('h').compute()
    vzy.name = 'Merid height adv'
    vzy.math = r'$\hat{v}\bar{\zeta}_{\tilde{y}}$'
    return vzy


def get_wds(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        uzx = get_zonal_zeta_advds(ds1)
        vzy = get_merid_zeta_advds(ds1)
        wd = ds1.wd.zep().read().move_to('l').compute()
    w = (wd + uzx + vzy).compute()
    w.name = 'Mean vert vel'
    w.math = r'$\bar{w}^z$'
    return w


def get_wds1(ds):
    uzx = get_zonal_zeta_advds(ds)
    vzy = get_merid_zeta_advds(ds)
    wd = ds.wd.zep().read().move_to('l').compute()
    w = (wd + uzx + vzy).compute()
    w.name = 'Mean vert vel'
    w.math = r'$\bar{w}^z$'
    return w


def get_velocities(fil1, **initializer):
    initializer['final_loc'] = 'hl'
    with pym6.Dataset(fil1, **initializer) as ds1:
        v = ds1.v.ysm().read().move_to('h').compute(check_loc=False)
        v.name = 'Mean merid vel'
        v.math = r'$\bar{v}^z$'

        u = ds1.u.xsm().read().move_to('h').compute(check_loc=False)
        u.name = 'Mean zonal vel'
        u.math = r'$\bar{u}^z$'

        w = get_wds1(ds1)

    ret_dict = dict(v=v, u=u, w=w)
    z = initializer.get('z', None)
    if z is not None:
        with pym6.Dataset(fil1, **initializer) as ds1:
            e = ds1.e.final_loc('hi').read().compute()
    meanax = initializer.get('meanax', None)
    tokm = initializer.get('tokm', 3)
    for key, value in ret_dict.items():
        if z is not None:
            value = value.toz(z, e)
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


def plot_one_panel(key, ax, alphabet, returned_dict, fil1, **initializer):
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
        with pym6.Dataset(fil1, **initializer) as ds1:
            e = ds1.e.zep().read().compute()
        if meanax is not None:
            e = e.nanmean(axis=meanax)
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


def plot_velocities(dict_or_func, fil1, **initializer):
    if callable(dict_or_func):
        returned_dict = dict_or_func(fil1, **initializer)
    else:
        returned_dict = dict_or_func

    figsize = initializer.get('figsize', (5, 5))
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)

    plot_one_panel('u', ax[0], 'a', returned_dict, fil1, **initializer)
    plot_one_panel('v', ax[1], 'b', returned_dict, fil1, **initializer)
    plot_one_panel('w', ax[2], 'c', returned_dict, fil1, **initializer)
    return fig
