from netCDF4 import Dataset as dset, MFDataset as mfdset
import numpy as np
import xarray as xr
from pymom6.pymom6 import _get_rho_at_z
from numba import jit, float32, float64
import comparison as cp
import sys
import pickle


def load_data():
    import pickle
    f = open('detected_eddies_surface_37_49_delvars', mode='rb')
    eddies, _ = pickle.load(f)
    f.close()
    eddies.erlist = eddies.erlist[:-3]
    bg = Background(cp.control)
    for eddy in eddies.iter_eddy():
        bg.all(eddy)
        sys.stdout.write(f"\r{eddy.id_}")
        sys.stdout.flush()
    eddies1 = eddies.erlist[:len(eddies) // 4]
    eddies2 = eddies.erlist[len(eddies) // 4:len(eddies) // 2]
    eddies3 = eddies.erlist[len(eddies) // 2:3 * len(eddies) // 4]
    eddies4 = eddies.erlist[3 * len(eddies) // 4:]
    f = open('detected_eddies_surface_37_49_pprofs_tsecs1', mode='wb')
    pickle.dump(eddies1, f)
    f.close()
    f = open('detected_eddies_surface_37_49_pprofs_tsecs2', mode='wb')
    pickle.dump(eddies2, f)
    f.close()
    f = open('detected_eddies_surface_37_49_pprofs_tsecs3', mode='wb')
    pickle.dump(eddies3, f)
    f.close()
    f = open('detected_eddies_surface_37_49_pprofs_tsecs4', mode='wb')
    pickle.dump(eddies4, f)
    f.close()
    return eddies


class Indices():
    def __init__(self, xnearidx, xsecidx, xnear, ynearidx, ysecidx, ynear):
        self.xnearidx = xnearidx
        self.xnear = xnear
        self.xsecidx = xsecidx
        self.ynearidx = ynearidx
        self.ynear = ynear
        self.ysecidx = ysecidx


class Background():
    def __init__(self, expt, **kwargs):
        fh = mfdset(expt.fil0)
        self.x = fh.variables['xh'][:]
        self.y = fh.variables['yh'][:]
        self.zi = fh.variables['zi'][:]
        self.zl = fh.variables['zl'][:]
        self.z = kwargs.get('z', np.linspace(-3000, 0))
        self.e = fh.variables['e'][:]
        self.fh.close()

        fhmean = mfdset(expt.fil2)
        self.emean = np.mean(fhmean.variables['e'][:], axis=0, keepdims=True)
        fhmean.close()

    # def close(self):
    #     self.fh.close()

    def nearest(self, eddy):
        ynearidx, ynear = self.find_closest_value(self.y, eddy.yc)
        xnearidx, xnear = self.find_closest_value(self.x, eddy.xc)
        xsecidx = np.nonzero((self.x > eddy.xc - 1) &
                             (self.x < eddy.xc + 1))[0]
        ysecidx = np.nonzero((self.y > eddy.yc - 1) &
                             (self.y < eddy.yc + 1))[0]
        return Indices(xnearidx, xsecidx, xnear, ynearidx, ysecidx, ynear)

    def xinterfaces(self, eddy):
        idx = self.nearest(eddy)
        esec = self.e[eddy.time, :, idx.ynearidx, idx.xsecidx].squeeze()
        return xr.DataArray(
            esec, coords=[('zi', self.zi), ('Lon', self.x[idx.xsecidx])])

    def xheights(self, eddy, anom=True):
        idx = self.nearest(eddy)
        hsec = -np.diff(
            self.e[eddy.time, :, idx.ynearidx, idx.xsecidx].squeeze(), axis=0)
        if anom:
            hsecmean = -np.diff(
                self.emean[:, :, idx.ynearidx, idx.xsecidx].squeeze(), axis=0)
            hsec = (hsec - hsecmean) / hsecmean
        return xr.DataArray(
            hsec, coords=[('zl', self.zl), ('Lon', self.x[idx.xsecidx])])

    def pressure(self, eddy, anom=True):
        idx = self.nearest(eddy)
        e = self.e[eddy.time, :, idx.ynearidx, idx.xnearidx].squeeze()
        p = np.zeros(self.zi.size)
        p[1:] = np.cumsum(-self.zl * np.diff(e) * 9.8)
        p = np.interp(self.z, e[::-1], p[::-1])
        if anom:
            emean = self.emean[:, :, idx.ynearidx, idx.xnearidx].squeeze()
            pmean = np.zeros(self.zi.size)
            pmean[1:] = np.cumsum(-self.zl * np.diff(emean) * 9.8)
            pmean = np.interp(self.z, emean[::-1], pmean[::-1])
            p = p - pmean
        p = p - np.mean(p)
        return p

    def pressurenodes(self, eddy):
        p = eddy.p
        zcp = p[:-1] * p[1:]
        zcp = np.size(zcp[zcp < 0])
        return zcp

    def temperaturesec1(self, eddy, anom=True, tbot=5, rho0=1031, drhodt=-0.2):
        idx = self.nearest(eddy)
        esec = self.e[eddy.time:eddy.time + 1, :, idx.ynearidx:idx.ynearidx +
                      1, idx.xsecidx]
        tsec = tbot + (self.get_rho_at_z(
            esec.astype(np.float64), self.z.astype(np.float64),
            self.zl.astype(np.float64), float(0)) - rho0) / drhodt
        if anom:
            eallsec = self.e[:, :, idx.ynearidx:idx.ynearidx + 1, idx.xsecidx]
            tallsec = tbot + (self.get_rho_at_z(
                eallsec.astype(np.float64), self.z.astype(np.float64),
                self.zl.astype(np.float64), float(0)) - rho0) / drhodt
            tsec = tsec - np.mean(tallsec, axis=0, keepdims=True)
        return xr.DataArray(
            tsec.squeeze(),
            coords=[('z', self.z), ('Lon', self.x[idx.xsecidx])])

    def pressure1(self, eddy, anom=True):
        idx = self.nearest(eddy)
        e = self.e[eddy.time, :, idx.ynearidx, idx.xnearidx].squeeze()
        p = np.zeros(self.zi.size)
        p[1:] = np.cumsum(-self.zl * np.diff(e) * 9.8)
        p = np.interp(self.z, e[::-1], p[::-1])
        if anom:
            e = self.e[:, :, idx.ynearidx, idx.xnearidx].squeeze()
            pmean = np.zeros(e.shape)
            pmean1 = np.zeros((e.shape[0], self.z.size))
            for i in range(e.shape[0]):
                pmean[i, 1:] = np.cumsum(-self.zl * np.diff(e[i, :]) * 9.8)
                pmean1[i, :] = np.interp(self.z, e[i, ::-1], pmean[i, ::-1])
            pmean1 = np.mean(pmean1, axis=0)
            p = p - pmean1
        p = p - np.mean(p)
        return p

    def all(self, eddy, anom=True, tbot=5, rho0=1031, drhodt=-0.2):
        idx = self.nearest(eddy)
        e = self.e[eddy.time, :, idx.ynearidx, idx.xnearidx].squeeze()
        p = np.zeros(self.zi.size)
        p[1:] = np.cumsum(-self.zl * np.diff(e) * 9.8)
        p = np.interp(self.z, e[::-1], p[::-1])
        if anom:
            e = self.e[:, :, idx.ynearidx, idx.xnearidx].squeeze()
            pmean = np.zeros(e.shape)
            pmean1 = np.zeros((e.shape[0], self.z.size))
            for i in range(e.shape[0]):
                pmean[i, 1:] = np.cumsum(-self.zl * np.diff(e[i, :]) * 9.8)
                pmean1[i, :] = np.interp(self.z, e[i, ::-1], pmean[i, ::-1])
            pmean1 = np.mean(pmean1, axis=0)
            p = p - pmean1
        p = p - np.mean(p)
        zcp = p[:-1] * p[1:]
        zcp = np.size(zcp[zcp < 0])
        eddy.p = p
        eddy.zcp = zcp
        esec = self.e[eddy.time:eddy.time + 1, :, idx.ynearidx:idx.ynearidx +
                      1, idx.xsecidx[0]:idx.xsecidx[-1] + 1]
        tsec = tbot + (self.get_rho_at_z(
            esec.astype(np.float64), self.z.astype(np.float64),
            self.zl.astype(np.float64), float(0)) - rho0) / drhodt
        if anom:
            eallsec = self.e[:, :, idx.ynearidx:idx.ynearidx +
                             1, idx.xsecidx[0]:idx.xsecidx[-1] + 1]
            tallsec = tbot + (self.get_rho_at_z(
                eallsec.astype(np.float64), self.z.astype(np.float64),
                self.zl.astype(np.float64), float(0)) - rho0) / drhodt
            tsec = tsec - np.mean(tallsec, axis=0, keepdims=True)
        eddy.tsec = xr.DataArray(
            tsec.squeeze(),
            coords=[('z', self.z),
                    ('Lon', self.x[idx.xsecidx[0]:idx.xsecidx[-1] + 1])])
        eddy.xinterfaces = xr.DataArray(
            esec.squeeze(),
            coords=[('zi', self.zi),
                    ('Lon', self.x[idx.xsecidx[0]:idx.xsecidx[-1] + 1])])

    @staticmethod
    def find_closest_value(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    get_rho_at_z = staticmethod(
        jit(float64[:, :, :, :](float64[:, :, :, :], float64[:], float64[:],
                                float32))(_get_rho_at_z))

    def temperaturesec(self, eddy, anom=True, tbot=5, rho0=1031, drhodt=-0.2):
        idx = self.nearest(eddy)
        esec = self.e[eddy.time:eddy.time + 1, :, idx.ynearidx:idx.ynearidx +
                      1, idx.xsecidx]
        tsec = tbot + (self.get_rho_at_z(
            esec.astype(np.float64), self.z.astype(np.float64),
            self.zl.astype(np.float64), float(0)) - rho0) / drhodt
        if anom:
            emeansec = self.emean[:, :, idx.ynearidx:idx.ynearidx +
                                  1, idx.xsecidx]
            tmeansec = tbot + (self.get_rho_at_z(
                emeansec.astype(np.float64), self.z.astype(np.float64),
                self.zl.astype(np.float64), float(0)) - rho0) / drhodt
            tsec = tsec - tmeansec
        return xr.DataArray(
            tsec.squeeze(),
            coords=[('z', self.z), ('Lon', self.x[idx.xsecidx])])
