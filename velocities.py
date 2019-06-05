import pymom6.pymom6 as pym6
import numpy as np
import importlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import string
# import matplotlib.gridspec as gspec
importlib.reload(pym6)


def get_zonal_zeta_adv(fil1, fil2, **initializer):
    initializer['final_loc'] = 'hl'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h = ds1.h_Cu.xsm().read().nanmean(axis=0).compute(check_loc=False)
        a = h.values
        a[a < htol] = np.nan
        h.values = a
        uh = ds2.uh.xsm().read().nanmean(axis=0).divide_by('dyCu').compute(
            check_loc=False)
        ur = (uh / h).compute(check_loc=False)
        a = ur.values
        a[np.isnan(a)] = 0
        ur.values = a
        zx = ds2.e.zep().xsm().xep().read().dbyd(3).nanmean(
            axis=0).move_to('l').compute(check_loc=False)
        uzx = (ur * zx).move_to('h').compute()
        uzx.name = 'Zonal height adv'
        uzx.math = r'$\hat{u}\bar{\zeta}_{\tilde{x}}$'
    return uzx


def get_merid_zeta_adv(fil1, fil2, **initializer):
    initializer['final_loc'] = 'hl'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        h = ds1.h_Cv.ysm().read().nanmean(axis=0).compute(check_loc=False)
        a = h.values
        a[a < htol] = np.nan
        h.values = a
        vh = ds2.vh.ysm().read().nanmean(axis=0).divide_by('dxCv').compute(
            check_loc=False)
        vr = (vh / h).compute(check_loc=False)
        a = vr.values
        a[np.isnan(a)] = 0
        vr.values = a
        zy = ds2.e.zep().ysm().yep().read().dbyd(2).nanmean(
            axis=0).move_to('l').compute(check_loc=False)
        vzy = (vr * zy).move_to('h').compute()
        vzy.name = 'Merid height adv'
        vzy.math = r'$\hat{v}\bar{\zeta}_{\tilde{y}}$'
    return vzy


def get_merid_bol_vel(fil1, fil2, **initializer):
    initializer['final_loc'] = 'hl'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        v = ds2.v.ysm().read().move_to('h').nanmean(axis=0).compute(
            check_loc=False)

        h = ds1.h_Cv.ysm().read().nanmean(axis=0).compute(check_loc=False)
        a = h.values
        a[a < htol] = np.nan
        h.values = a
        vh = ds2.vh.ysm().read().nanmean(axis=0).divide_by('dxCv').compute(
            check_loc=False)
        vr = (vh / h).move_to('h').compute()
        a = vr.values
        a[np.isnan(a)] = 0
        vr.values = a

        vbolus = (vr - v).compute()
        vbolus.name = 'Eddy-induced merid vel'
        vbolus.math = r'$\frac{\overline{\sigma^\prime v^\prime}}{\bar{\sigma}}$'
    return vbolus


def get_zonal_bol_vel(fil1, fil2, **initializer):
    initializer['final_loc'] = 'hl'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        u = ds2.u.xsm().read().move_to('h').nanmean(axis=0).compute(
            check_loc=False)

        h = ds1.h_Cu.xsm().read().nanmean(axis=0).compute(check_loc=False)
        a = h.values
        a[a < htol] = np.nan
        h.values = a
        uh = ds2.uh.xsm().read().nanmean(axis=0).divide_by('dyCu').compute(
            check_loc=False)
        ur = (uh / h).move_to('h').compute()
        a = ur.values
        a[np.isnan(a)] = 0
        ur.values = a

        ubolus = (ur - u).compute()
        ubolus.name = 'Eddy-induced zonal vel'
        ubolus.math = r'$\frac{\overline{\sigma^\prime u^\prime}}{\bar{\sigma}}$'
    return ubolus


def get_whash(fil1, fil2, **initializer):
    uzx = get_zonal_zeta_adv(fil1, fil2, **initializer)
    vzy = get_merid_zeta_adv(fil1, fil2, **initializer)
    with pym6.Dataset(fil2, **initializer) as ds2:
        wd = ds2.wd.zep().read().nanmean(axis=0).move_to('l').compute()
        wd = (-wd).compute()
    whash = (wd + uzx + vzy).compute()
    whash.name = 'Residual vert vel'
    whash.math = r'$w^{\sharp}$'
    return whash


def get_velocities(fil1, fil2, **initializer):
    initializer['final_loc'] = 'hl'
    htol = initializer.get('htol', 1e-3)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        v = ds2.v.ysm().read().move_to('h').nanmean(axis=0).compute(
            check_loc=False)
        v.name = 'Mean merid vel'
        v.math = r'$\bar{v}$'

        h = ds1.h_Cv.ysm().read().nanmean(axis=0).compute(check_loc=False)
        a = h.values
        a[a < htol] = np.nan
        h.values = a
        vh = ds2.vh.ysm().read().nanmean(axis=0).divide_by('dxCv').compute(
            check_loc=False)
        vr = (vh / h).move_to('h').compute()
        a = vr.values
        a[np.isnan(a)] = 0
        vr.values = a
        vr.name = 'Residual merid vel'
        vr.math = r'$\hat{v}$'

        u = ds2.u.xsm().read().move_to('h').nanmean(axis=0).compute(
            check_loc=False)
        u.name = 'Mean zonal vel'
        u.math = r'$\bar{u}$'

        h = ds1.h_Cu.xsm().read().nanmean(axis=0).compute(check_loc=False)
        a = h.values
        a[a < htol] = np.nan
        h.values = a
        uh = ds2.uh.xsm().read().nanmean(axis=0).divide_by('dyCu').compute(
            check_loc=False)
        ur = (uh / h).move_to('h').compute()
        a = ur.values
        a[np.isnan(a)] = 0
        ur.values = a
        ur.name = 'Residual zonal vel'
        ur.math = r'$\hat{u}$'

        wd = ds2.wd.zep().read().nanmean(axis=0).move_to('l').compute()
        wd = (-wd).compute()
        wd.name = 'Dia comp of vert vel'
        wd.math = r'$\hat{\varpi}\bar{\zeta}_{\tilde{b}}$'

    uzx = get_zonal_zeta_adv(fil1, fil2, **initializer)
    vzy = get_merid_zeta_adv(fil1, fil2, **initializer)
    adiaw = (uzx + vzy).compute()
    adiaw.name = 'Epi comp of vert vel'
    adiaw.math = uzx.math + '$+$' + vzy.math
    whash = get_whash(fil1, fil2, **initializer)
    ubolus = get_zonal_bol_vel(fil1, fil2, **initializer)
    vbolus = get_merid_bol_vel(fil1, fil2, **initializer)

    ret_dict = dict(
        v=v,
        vbolus=vbolus,
        vr=vr,
        u=u,
        ubolus=ubolus,
        ur=ur,
        wd=wd,
        adiaw=adiaw,
        whash=whash)
    # print_details(ret_dict)
    return ret_dict


def print_details(dic):
    for key, value in dic.items():
        print(value.name, value.shape)
        for key1, value1 in value.dimensions.items():
            print(key1, value1.size)


def plot_one_panel(key, ax, alphabet, returned_dict, fil2, **initializer):
    z = initializer.get('z', None)
    vmin = initializer.get('vmin', None)
    vmax = initializer.get('vmax', None)
    eskip = initializer.get('eskip', 2)
    textx = initializer.get('textx', 0.03)
    texty = initializer.get('texty', 0.1)
    fontsize = initializer.get('fontsize', 15)
    interpolation = initializer.get('interpolation', 'none')
    if z is not None:
        with pym6.Dataset(fil2, **initializer) as ds2:
            e = ds2.e.zep().read().nanmean(axis=(0, 2)).compute()
        data = returned_dict[key].nanmean(axis=2).toz(
            z, e).tokm(3).to_DataArray(check_loc=False)
        im = data.squeeze().plot.imshow(
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap='RdBu_r',
            interpolation=interpolation,
            add_colorbar=False,
            add_labels=False)
        if ax in ax.get_figure().axes[slice(-4, None, 1)]:
            ax.set_xlabel(list(data.coords.keys())[3])
        if ax in ax.get_figure().axes[slice(0, None, 3)]:
            ax.set_ylabel(list(data.coords.keys())[1])
        e = e.tokm(3).to_DataArray().squeeze()
        ax.plot(e.coords[e.dims[1]], e.values[::eskip, :].T, 'k', lw=1)
        # ax.set_title('(' + alphabet + ') ' + returned_dict[key].name)
        ax.set_title('(' + alphabet + ') ')
        ax.text(
            textx,
            texty,
            returned_dict[key].math,
            transform=ax.transAxes,
            fontsize=fontsize,
            bbox=dict(facecolor='w', edgecolor='w', alpha=0.8))
    return im


def plot_three_panels(key_list, ax_list, alphabet_list, returned_dict, fil2,
                      **initializer):
    for key, ax, alphabet in zip(key_list, ax_list, alphabet_list):
        im = plot_one_panel(key, ax, alphabet, returned_dict, fil2,
                            **initializer)
    cbar_ax, kw = mpl.colorbar.make_axes(ax_list)
    cb = plt.colorbar(im, cax=cbar_ax, **kw)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()


def plot_velocities(dict_or_func, fil1, fil2, **initializer):
    if callable(dict_or_func):
        returned_dict = dict_or_func(fil1, fil2, **initializer)
    else:
        returned_dict = dict_or_func

    figsize = initializer.get('figsize', (5, 5))
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=figsize)

    plot_three_panels(['u', 'ubolus', 'ur'],
                      ax[0, :].ravel().tolist(),
                      string.ascii_lowercase[:3],
                      returned_dict,
                      fil2,
                      vmin=-0.1,
                      vmax=0.1,
                      **initializer)
    plot_three_panels(['v', 'vbolus', 'vr'],
                      ax[1, :].ravel().tolist(),
                      string.ascii_lowercase[3:6],
                      returned_dict,
                      fil2,
                      vmin=-0.8,
                      vmax=0.8,
                      **initializer)
    plot_three_panels(['wd', 'adiaw', 'whash'],
                      ax[2, :].ravel().tolist(),
                      string.ascii_lowercase[6:9],
                      returned_dict,
                      fil2,
                      vmin=-4e-4,
                      vmax=4e-4,
                      **initializer)
    ax.ravel()[0].set_ylim((initializer.get('z', [-3000, -3000])[0], 0))
    return fig


#def extract_budget(fil1, fil2, fil3=None, **initializer):
#    meanax = initializer.get('meanax', 2)
#    initializer.pop('meanax')
#    z = initializer.get('z', None)
#    if 'z' in initializer:
#        initializer.pop('z')
#    initializer['final_loc'] = 'hi'
#    with pym6.Dataset(fil2, **initializer) as ds:
#        e = ds.e.zep().read().nanmean(axis=(0, 2)).compute()
#    initializer['final_loc'] = 'hl'
#    if fil3 is not None:
#        with pym6.Dataset(fil3) as ds:
#            islaydeepmax = ds.islayerdeep.read().compute(check_loc=False).array
#            islaydeepmax = islaydeepmax[0, 0, 0, 0]
#        with pym6.Dataset(
#                fil3, fillvalue=np.nan, **initializer) as ds, pym6.Dataset(
#                    fil1, **initializer) as ds2:
#            swash = ds.islayerdeep
#            uv = ds2.uv
#            swash.indices = uv.indices
#            swash.dim_arrays = uv.dim_arrays
#            swash = swash.ysm().xsm().read().move_to('u').move_to('h').nanmean(
#                axis=(0, 2)).compute()
#            swash = ((-swash + islaydeepmax) / islaydeepmax * 100).compute()
#            if z is not None:
#                swash = swash.toz(z, e)
#            swash = swash.tokm(3).to_DataArray()
#    else:
#        swash = None
#    blist = get_thickness_budget(fil1, fil2, **initializer)
#    namelist = []
#    for i, b in enumerate(blist):
#        namelist.append(b.name)
#        b = b.nanmean(axis=meanax)
#        if z is not None:
#            b = b.toz(z, e)
#        blist[i] = b.tokm(3).to_DataArray()
#    blist_concat = xr.concat(blist, dim=pd.Index(namelist, name='Term'))
#    blist_concat.name = 'Thickness Budget'
#    e = e.tokm(3).to_DataArray()
#    return_dict = dict(
#        blist_concat=blist_concat, blist=blist, e=e, swash=swash)
#    return return_dict
