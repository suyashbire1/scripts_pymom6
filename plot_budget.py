import string
import numpy as np
import matplotlib.pyplot as plt


def plot_budget(func_or_dict, fil1, fil2, **initializer):
    vmin = initializer.get('vmin', -8e-5)
    vmax = initializer.get('vmax', 8e-5)
    col_wrap = initializer.get('col_wrap')
    size = initializer.get('size', 3)
    aspect = initializer.get('aspect', (1 + np.sqrt(5)) / 2)
    yincrease = initializer.get('yincrease', True)
    textx = initializer.get('textx', 0.03)
    texty = initializer.get('texty', 0.1)
    fontsize = initializer.get('fontsize', 15)
    eskip = initializer.get('eskip', 2)
    elevs = initializer.get(
        'elevs', [-2500, -1500, -1000, -800, -500, -300, -200, -100, -50, 0])
    toz = initializer.get('toz', True)
    add_colorbar = initializer.get('add_colorbar', True)
    cbar_kwargs = initializer.get('cbar_kwargs', {})
    interpolation = initializer.get('interpolation', 'none')
    if callable(func_or_dict):
        returned_dict = func_or_dict(fil1, fil2, **initializer)
    else:
        returned_dict = func_or_dict
    blist_concat = returned_dict['blist_concat']
    blist = returned_dict['blist']
    e = returned_dict['e']
    if e is not None:
        e = e.squeeze()
    PV = returned_dict.get('PV', None)
    swash = returned_dict.get('swash')
    if PV is not None:
        PV = PV.squeeze()
        PVlevels = initializer.get('PVlevels', np.logspace(-6, -5.5, 5))
    if swash is not None:
        swash = swash.squeeze()
        swashperc = initializer.get('swashperc', 1)
    blist_concat.values = np.ma.masked_invalid(blist_concat.values)
    blist_concat.name = initializer.get('Name', '')
    fg = blist_concat.squeeze().plot.imshow(
        size=size,
        aspect=aspect,
        yincrease=yincrease,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        cmap='RdBu_r',
        col='Term',
        col_wrap=col_wrap,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs)
    name = initializer.get('name', True)
    for i, b in enumerate(blist):
        ax = fg.axes.flat[i]
        if name:
            ax.set_title('(' + string.ascii_lowercase[i] + ') ' + b.name)
        else:
            ax.set_title('(' + string.ascii_lowercase[i] + ') ')
        ax.text(
            textx,
            texty,
            b.attrs['math'],
            transform=ax.transAxes,
            fontsize=fontsize,
            bbox=dict(facecolor='w', edgecolor='w', alpha=0.8))
        if e is not None:
            if toz:
                ax.plot(e.coords[e.dims[1]], e.values[::eskip, :].T, 'k', lw=1)
            else:
                cs = e.plot.contour(
                    ax=ax, levels=elevs, colors='k', linewidths=1)
                plt.clabel(
                    cs,
                    # levels[1::2],  # label every second level
                    inline=1,
                    fmt='%1.1f',
                    fontsize=12)
        if PV is not None:
            ax.contour(
                PV.coords[PV.dims[1]],
                PV.coords[PV.dims[0]],
                PV.values * 1e6,
                colors='slategray',
                levels=PVlevels,
                linewidths=2).clabel(
                    inline=True, fmt="%.2f")
            pass
        if swash is not None:
            ax.contour(
                swash.coords[swash.dims[1]],
                swash.coords[swash.dims[0]],
                swash.values,
                np.array([swashperc]),
                colors='forestgreen',
                linewidths=2)

    fg.cbar.formatter.set_powerlimits((0, 0))
    fg.cbar.update_ticks()
    return fg
