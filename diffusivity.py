import numpy as np
from scipy.optimize import curve_fit
import twamomy_budget as ty
import energetics as ee
import pymom6.pymom6 as pym6


def slope(x, a):
    return a * x


def slopefit(x, y, **kwargs):
    """Fits y on x using slope

    :param x: x (numpy array)
    :param y: y (numoy array)
    :returns: Tuple containing popt and stddev of popt
    :rtype: tuple

    """
    n = len(y)
    popt, _ = curve_fit(slope, x, y, **kwargs)
    yfit = slope(x, *popt)
    SSE = np.sum((yfit - y)**2)
    se = np.sqrt(SSE / (n - 2))
    stdevb = se / np.sqrt(np.sum((x - np.mean(x))**2))
    return popt[0], stdevb


def default_xew():
    """Returns default xew"""

    a = np.arange(-20, -1) / 2
    b = -np.array([0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
    xew = np.hstack((a, b))
    return xew


def profile_ke(h_x, fd, xew=None):
    """Returns zonal diffusivity profile at xew locations

    :param h_x: h_x
    :param fd: form drag
    :param xew: edges of the bins
    :returns: tuple of bin locations, slope, and slope stdevs
    :rtype: tuple

    """
    if xew is None:
        xew = default_xew()
    xp, yp, yerr = [], [], []
    for i in range(xew.size - 1):
        hxsub = h_x.sel(xh=slice(xew[i], xew[i + 1])).isel(
            zl=slice(None, -1)).values.ravel() * 1e3
        fdsub = fd.sel(xh=slice(xew[i], xew[i + 1])).isel(
            zl=slice(None, -1)).values.ravel() * 1e3
        popt, stdev = slopefit(hxsub, fdsub)
        xp.append(0.5 * (xew[i] + xew[i + 1]))
        yp.append(popt)
        yerr.append(3 * stdev)
    return xp, yp, yerr


def profile_eke(eke, vsq, xew=None):
    """Returns zonal profile of eke and vsq at bin centers

    :param eke: eke
    :param vsq: vsq
    :param xew: edges of the bins
    :returns: tuple of eke and vsq at bin centers
    :rtype: tuple

    """
    if xew is None:
        xew = default_xew()
    ekep, vsqp = [], []
    for i in range(xew.size - 1):
        ekesub = eke.sel(xh=slice(xew[i], xew[i + 1])).mean().values
        vsqsub = vsq.sel(xh=slice(xew[i], xew[i + 1])).mean().values
        ekep.append(ekesub)
        vsqp.append(vsqsub)
    return ekep, vsqp


def eke_fit(x, a, b):
    return a / (1 + b * x[1]) * x[0]


def get_eke_fit_params(x, y, **kwargs):
    """Fits y on x using eke_fit

    :param x: x (numpy array)
    :param y: y (numoy array)
    :returns: Tuple containing popt and stddev of popt
    :rtype: tuple

    """
    n = len(y)
    popt, _ = curve_fit(eke_fit, x, y, **kwargs)
    kefit = eke_fit(x, *popt)
    SSE = np.sum((kefit - y)**2)
    se = np.sqrt(SSE / (n - 2))
    stdevb = se / np.sqrt(np.sum((x - np.mean(x))**2))
    return popt[0], stdevb


def get_sigFD(fil1, fil2, **initializer):
    initializer['final_loc'] = 'vl'
    edlsqmy = ty.get_edlsqmy(fil1, fil2, **initializer)
    h = ty.get_h(fil1, fil2, **initializer)
    with pym6.Dataset(fil1, **initializer) as ds1, pym6.Dataset(
            fil2, **initializer) as ds2:
        hpfv = ds1.twa_hpfv.read().nanmean(axis=0).compute()
        pfvm = ds2.PFv.read().nanmean(axis=0).compute()
        db = np.diff(ds1.zl)[0] * 9.8 / 1000
        bdivep4 = -((-hpfv).compute() + (h * pfvm).compute() -
                    (edlsqmy * db * 0.5).compute()).compute()
    bdivep4.name = 'Form drag'
    bdivep4.math = (r"""$-(\overline{\zeta ^\prime m_{\tilde{y}}^\prime})"""
                    r"""_{\tilde{b}}$""")
    bdivep4.units = r'ms$^{-2}$'
    return bdivep4


def get_vsq(fil1, fil2, **initializer):
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        h = ds1.h_Cv.read().nanmean(axis=(0, 2)).compute()
        vsq = ds.v.read().nanmean(axis=(0, 2)).compute()
        vsq.values[h.values < initializer['htol']] = 0
        vsq = vsq.to_DataArray()**2
        vsq = vsq.mean('zl')
    return vsq


def get_hx(fil1, fil2, **initializer):
    bc = dict(
        v=['mirror', 'circsymh', 'zeros', 'circsymq', 'circsymh', 'neumann'])
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        hx = ds1.h_Cv.bc_type(bc).xsm().xep().read().dbyd(3).move_to(
            'v').nanmean(axis=(0, 2)).compute()
        h = ds1.h_Cv.read().nanmean(axis=(0, 2)).compute()
        hx.values[h.values < initializer['htol']] = 0
    return h, hx


def plot_kappa_with_eke(expt,
                        col,
                        ax,
                        ax1,
                        ax2,
                        south_lat,
                        north_lat,
                        west_lon,
                        east_lon,
                        xew=None):
    initializer = dict(
        south_lat=south_lat,
        north_lat=north_lat,
        west_lon=west_lon,
        east_lon=east_lon,
        geometry=expt.geometry,
        htol=1e1)
    omega = 2 * np.pi / 24 / 3600
    f = 2 * omega * np.sin(np.radians(0.5 * (south_lat + north_lat)))
    sigfd = get_sigFD(expt.fil1, expt.fil2, **initializer)
    sigfd = sigfd.nanmean(axis=2).compute()
    eke = ee.eke(expt.fil1, expt.fil2, final_loc='hl', **initializer)
    eke = (eke.nanmean(axis=2).to_DataArray().sum('zl') / 3000)  #**0.5
    vsq = get_vsq(expt.fil1, expt.fil2, **initializer)
    h, hx = get_hx(expt.fil1, expt.fil2, **initializer)
    sigfd.values[h.values < initializer['htol']] = 0
    if xew is None:
        xew = default_xew()
    xp, ke, keerr = profile_ke(
        hx.to_DataArray(), sigfd.to_DataArray(), xew=xew)
    ekep, vsqp = profile_eke(eke, vsq, xew=xew)
    ke = -np.array(ke) / f
    keerr = -np.array(keerr) / f

    popt, _ = curve_fit(
        eke_fit,
        np.array([ekep, vsqp]),
        ke,
        bounds=([-np.inf, 0], [np.inf, np.inf]))
    kefit = eke_fit(np.array([ekep, vsqp]), *popt)
    ax.plot(xp, kefit / 1e4, label=expt.name, color=col)
    ax.errorbar(xp, ke / 1e4, yerr=keerr / 1e4, color=col, fmt='o')
    ax.set_ylabel(r'$K_e$ ($10^{4}$ m$^2$s$^{-1}$)')
    ax.set_xlabel(r'x ($^{\circ}$)')
    ax.grid()
    ax1.plot(xp, ekep, label=expt.name, color=col)
    ax1.set_ylabel(r'EKE (m$^2$s$^{-2}$)')
    ax1.set_xlabel(r'x ($^{\circ}$)')
    ax1.grid()
    ax2.plot(xp, vsqp, label=expt.name, color=col)
    ax2.set_ylabel(r'$v^2$ (m$^2$s$^{-2}$)')
    ax2.set_xlabel(r'x ($^{\circ}$)')
    ax2.grid()
    return np.array(xp), ke, popt


def profile_eke_sig_zxzy(eke, sig, zx, zy, xew=None):
    """Returns zonal profile of eke, sigma, z_x, and z_y at bin centers

    :param eke: eke
    :param sig: sigma
    :param zx: zx
    :param zy: zy
    :param xew: edges of the bins
    :returns: tuple of eke, sigma, z_x, and z_y at bin centers
    :rtype: tuple

    """
    if xew is None:
        xew = default_xew()
    ekep, sigp, zxp, zyp = [], [], [], []
    for i in range(xew.size - 1):
        ekesub = eke.sel(xh=slice(xew[i], xew[i + 1])).mean().values
        sigsub = sig.sel(xh=slice(xew[i], xew[i + 1])).mean().values
        zxsub = zx.sel(xh=slice(xew[i], xew[i + 1])).mean().values
        zysub = zy.sel(xh=slice(xew[i], xew[i + 1])).mean().values
        ekep.append(ekesub)
        sigp.append(sigsub)
        zxp.append(zxsub)
        zyp.append(zysub)
    return ekep, sigp, zxp, zyp


def get_sigma(fil1, fil2, **initializer):
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        h = ds1.h_Cv.read().nanmean(axis=(0, 2)).to_DataArray()
        h.values[h.values < initializer['htol']] = 0
    return h.mean('zl')


def get_zxzy(fil1, fil2, **initializer):
    with pym6.Dataset(fil2, **initializer) as ds, pym6.Dataset(
            fil1, **initializer) as ds1:
        ex = ds.e.final_loc('hl').zep().xep().xsm().read().dbyd(3).move_to(
            'h').move_to('l').nanmean(axis=(0, 2)).to_DataArray()
        ey = ds.e.final_loc('hl').zep().yep().ysm().read().dbyd(2).move_to(
            'h').move_to('l').nanmean(axis=(0, 2)).to_DataArray()
    return (ex**2).mean('zl'), (ey**2).mean('zl')


def plot_kappa_with_geometric(expt,
                              col,
                              ax,
                              ax1,
                              ax2,
                              south_lat,
                              north_lat,
                              west_lon,
                              east_lon,
                              xew=None):
    initializer = dict(
        south_lat=south_lat,
        north_lat=north_lat,
        west_lon=west_lon,
        east_lon=east_lon,
        geometry=expt.geometry,
        htol=1e1)
    omega = 2 * np.pi / 24 / 3600
    f = 2 * omega * np.sin(np.radians(0.5 * (south_lat + north_lat)))
    sigfd = get_sigFD(expt.fil1, expt.fil2, **initializer)
    sigfd = sigfd.nanmean(axis=2).compute()
    eke = ee.eke(expt.fil1, expt.fil2, final_loc='hl', **initializer)
    eke = (eke.nanmean(axis=2).to_DataArray().sum('zl') / 3000)  #**0.5
    sigma = get_sigma(expt.fil1, expt.fil2, **initializer)
    zx, zy = get_zxzy(expt.fil1, expt.fil2, **initializer)
    h, hx = get_hx(expt.fil1, expt.fil2, **initializer)
    sigfd.values[h.values < initializer['htol']] = 0
    if xew is None:
        xew = default_xew()
    xp, ke, keerr = profile_ke(
        hx.to_DataArray(), sigfd.to_DataArray(), xew=xew)
    ekep, sigp, zxp, zyp = profile_eke_sig_zxzy(eke, sigma, zx, zy, xew=xew)
    ke = -np.array(ke) / f
    keerr = -np.array(keerr) / f
    predictor = np.array(ekep) * np.array(sigp)**0.5 / np.sqrt(
        np.array(zxp) + np.array(zyp))

    popt, _ = curve_fit(slope, predictor, ke)
    kefit = slope(predictor, *popt)
    ax.plot(xp, kefit / 1e4, label=expt.name, color=col)
    ax.errorbar(xp, ke / 1e4, yerr=keerr / 1e4, color=col, fmt='o')
    ax.set_ylabel(r'$K_e$ ($10^{4}$ m$^2$s$^{-1}$)')
    ax.set_xlabel(r'x ($^{\circ}$)')
    ax.grid()
    ax1.plot(xp, ekep, label=expt.name, color=col)
    ax1.set_ylabel(r'EKE (m$^2$s$^{-2}$)')
    ax1.set_xlabel(r'x ($^{\circ}$)')
    ax1.grid()
    ax2.plot(xp, zxp, label=expt.name, color=col, linestyle='-')
    ax2.plot(xp, zyp, color=col, linestyle='--')
    ax2.set_ylabel(r'$\zeta_x, \zeta_y$')
    ax2.set_xlabel(r'x ($^{\circ}$)')
    ax2.grid()
    return np.array(xp), ke, popt
