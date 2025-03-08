import gc, sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import healpy as hp
from scipy.optimize import curve_fit
from astropy.table import Table
import fitsio

from pixell import enmap, reproject, enplot, utils, wcsutils
from orphics import maps, mpi, io, stats, cosmology
import symlens

# Load a fiducial CMB theory object
theory = cosmology.default_theory()

pixwin = False # unsure
no_fit_noise = False
act_only_in_hres = False
curl = False # unsure

tap_per = 12.
pad_per = 3.
hres_lycut = 90
hres_lxcut = 50

highres_fit_ellmin = 500
highres_fit_ellmax = 8000

highres_fiducial_rms = 20.
highres_fiducial_lknee = 3000.
highres_fiducial_alpha = -4
gradient_fiducial_rms = 40.

gradient_fit_ellmin = 200
gradient_fit_ellmax = 3000

# stamp size and resolution
stamp_width_deg = 128. / 60.0        # stamp_width_arcmin: 128.0
pixel = 0.5                      # pix_width_arcmin: 0.5
maxr = stamp_width_deg * utils.degree / 2.0 # max radius for projection geometry 

# beam and FWHM
plc_beam_fwhm = 5.     # 5 arcmin
ilc_beam_fwhm = 1.6        # 1.6 arcmin

# Planck mask
xlmin = 200 ; xlmax = 2000     # 200, 2000

# ACT mask
ilcmin = 200 ; ilcmax = 8000     # 200, 8000
ylmin = 200 ; ylmax = 6000     # 200, 6000 -> 3500 

# kappa mask
klmin = 200 ; klmax = 5000             # 200, 5000 -> 3000

# for binned kappa profile
bin_edges = np.arange(0, 15, 1.5) # 15 arcmin, 1.5 arcmin
centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

# record planck
pmap = enmap.read_map('/pscratch/sd/b/boryanah/ACTxDESI/ACT/COM_CMB_IQU-smica-nosz_2048_R3.00_full_pixell.fits')

# load act map
amap_150 = enmap.read_map('/pscratch/sd/b/boryanah/ACTxDESI/ACT/hilc_fullRes_TT_17000.fits')

# load galaxy catalog and mask galaxies
cat_dir = "/pscratch/sd/b/boryanah/screening/dr9/"
PHOTO_ERR = 0.035
LOGM_MIN = 11.
data = np.load(cat_dir + f"sweep_LOGM{LOGM_MIN:.1f}_PHOTOZ{PHOTO_ERR:.1f}_masked.npz")
ras = data['ras']
decs = data['decs']
logms = data['logms']
LOGM_MIN = 12.5
mask = logms > LOGM_MIN
ras = ras[mask]
decs = decs[mask]
print("number of galaxies", len(ras))
del data
gc.collect()

# TODO: add the CAP filter perhaps

def fit_p1d(
        l_edges, cents, p1d, which, xout, bfunc1, bfunc2, rms=None, lmin=None, lmax=None, debug_fit=False
):
    # function for fitting 1D power spectrum of given stamp
    b1 = bfunc1 if bfunc1 is not None else lambda x: 1
    b2 = bfunc2 if bfunc2 is not None else lambda x: 1
    tfunc = lambda x: theory.lCl("TT", x) * b1(x) * b2(x)

    if no_fit_noise:
        # Use fiducial spectrum + RMS noise if no fitting requested
        x = xout
        ret = tfunc(x) + (rms * np.pi / 180.0 / 60.0) ** 2.0

        if debug_fit:
            pl = io.Plotter("Cell")
            pl.add(cents, p1d, ls="none", marker="o")
            pl.add(xout, ret, ls="none", marker="o")
            pl._ax.set_ylim(1e-7, 1)
            pl._ax.set_xlim(0, 6000)
            pl.done(f"fcl.png")
            sys.exit()

    else:
        # PS fitting
        # Select region for fit
        sel = np.logical_and(cents > lmin, cents < lmax)
        delta_ells = np.diff(l_edges)[sel]
        ells = cents[sel]
        cls = p1d[sel]
        cltt = tfunc(ells)  # fiducial Cltt
        if (which == "act" or which == "act_cross") and (act_only_in_hres):
            if which == "act" or which == "act_cross":
                # Get bandpower variance estimate based on cltt + fiducial 1/f + white noise
                w0 = highres_fiducial_rms
                sigma2 = stats.get_sigma2(
                    ells,
                    cltt,
                    w0,
                    delta_ells,
                    fsky,
                    ell0=highres_fiducial_lknee,
                    alpha=highres_fiducial_alpha,
                )
            func = stats.fit_cltt_power(
                ells,
                cls,
                tfunc,
                w0,
                sigma2,
                ell0=highres_fiducial_lknee,
                alpha=highres_fiducial_alpha,
                fix_knee=False,
            )
        elif (which == "plc") or (
            (which == "act" or which == "act_cross") and not (act_only_in_hres)
        ):
            w0 = gradient_fiducial_rms if which=='plc' else highres_fiducial_rms
            sigma2 = stats.get_sigma2(ells, cltt, w0, delta_ells, fsky, ell0=0, alpha=1)
            func = stats.fit_cltt_power(
                ells, cls, tfunc, w0, sigma2, ell0=0, alpha=1, fix_knee=True
            )
        elif which == "apcross":
            w0 = gradient_fiducial_rms
            w0p = highres_fiducial_rms
            ell0 = 0
            ell0p = highres_fiducial_lknee if (act_only_in_hres) else 0
            sigma2 = stats.get_sigma2(
                ells,
                cltt,
                w0,
                delta_ells,
                fsky,
                ell0=ell0,
                alpha=0,
                w0p=w0p,
                ell0p=ell0p,
                alphap=highres_fiducial_alpha if (act_only_in_hres) else 1,
                clxx=cltt,
                clyy=cltt,
            )
            func = stats.fit_cltt_power(
                ells,
                cls,
                tfunc,
                w0,
                sigma2,
                ell0=ell0p,
                alpha=highres_fiducial_alpha if (act_only_in_hres) else 1,
                fix_knee=True if not (act_only_in_hres) else False,
            )

        ret = func(xout)

        if debug_fit:
            pl = io.Plotter("Dell")
            ls = np.arange(10000)
            pl.add(ls, func(ls))
            pl.add(ls, tfunc(ls), ls="--")
            pl.add_err(
                cents[sel], p1d[sel], yerr=np.sqrt(sigma2), ls="none", marker="o"
            )
            pl._ax.set_ylim(1e-1, 1e5)
            pl.done(f"fcl.png")
            sys.exit()

    ret[xout < 2] = 0
    assert np.all(np.isfinite(ret))
    return ret


# HERE WE START LOOPING
for i in range(len(decs)):
    if i % 100 == 0: print(i)
    coords = np.array([decs[i], ras[i]]) * utils.degree

    """ 
    !! CUT OUT 150 STAMP
    """
    # cut out a stamp from the ACT map (CAR -> tan: gnomonic projection)
    astamp_150 = reproject.thumbnails(
        amap_150,
        coords,
        r=maxr,
        res=pixel * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=True
    )

    if np.any(astamp_150 >= 1e3):
        continue

    # cut out a stamp from the Planck map (CAR -> tangent)
    pstamp = reproject.thumbnails(
        pmap, coords, r=maxr, res=pixel * utils.arcmin, proj="tan", oversample=2, pixwin=False
    )[0]

    """ 
    !! COSINE TAPER
    """
    if i == 0:
        # get an edge taper map and apodize
        taper = maps.get_taper(
            astamp_150.shape,
            astamp_150.wcs,
            taper_percent=tap_per,
            pad_percent=pad_per,
            weight=None,
        )
        taper = taper[0]

    # applying this to the stamp makes it have a nice zeroed edge!
    act_stamp_150 = astamp_150 * taper
    plc_stamp = pstamp * taper

    """ 
    !! STAMP FFTs
    """
    k150 = enmap.fft(act_stamp_150, normalize="phys")
    kp = enmap.fft(plc_stamp, normalize="phys")
    
    if i == 0:

        """ 
        !! INITIALIZE CALCULATIONS BASED ON GEOMETRY
        """
        shape = astamp_150.shape
        wcs = astamp_150.wcs
        modlmap = enmap.modlmap(shape, wcs)
        
        # High-res beam functions
        bfunc150 = lambda x: maps.gauss_beam(ilc_beam_fwhm, x)

        # evaluate the 2D Gaussian beam on an isotropic Fourier grid
        act_150_kbeam2d = bfunc150(modlmap)
        plc_kbeam2d = maps.gauss_beam(modlmap, plc_beam_fwhm)

        # get theory spectrum - this should be the lensed spectrum!
        ells = np.arange(8000)
        cltt = theory.lCl("TT", ells)

        ## interpolate ells and cltt 1D power spectrum specification
        ## isotropically on to the Fourier 2D space grid
        # build interpolated 2D Fourier CMB from theory and maps
        ucltt = maps.interp(ells, cltt)(modlmap)

        # bin size and range for 1D binned power spectrum
        minell = 2 * maps.minimum_ell(shape, wcs)
        l_edges = np.arange(minell / 2, 8001, minell)
        lbinner = stats.bin2D(modlmap, l_edges)
        
	# PS correction factor
        w2 = np.mean(taper ** 2)
        
        # fsky for bandpower variance
        fsky = enmap.area(shape, wcs) * w2 / 4.0 / np.pi

        # build Fourier space masks for lensing reconstruction
        xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax)
        ymask = maps.mask_kspace(
            shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=hres_lxcut, lycut=hres_lycut
        ) 
        kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax)

        # map of distances from center
        modrmap = enmap.modrmap(shape, wcs)

    # Fourier map -> PS
    pow = lambda x, y: (x * y.conj()).real 
    
    # measure the binned power spectrum from given stamp
    act_cents, act_p1d_150 = lbinner.bin(pow(k150, k150) / w2)

    plc_cents, plc_p1d = lbinner.bin(pow(kp, kp) / w2)
    

    """ 
    !! FIT POWER SPECTRA
    """
    tclaa_150 = fit_p1d(
        l_edges,
        act_cents,
        act_p1d_150,
        "act",
        modlmap,
        bfunc150,
        bfunc150,
        rms=highres_fiducial_rms,
        lmin=highres_fit_ellmin,
        lmax=highres_fit_ellmax,
    )

    tclpp = fit_p1d(
        l_edges,
        plc_cents,
        plc_p1d,
        "plc",
        modlmap,
        lambda x: maps.gauss_beam(x, plc_beam_fwhm),
        lambda x: maps.gauss_beam(x, plc_beam_fwhm),
        rms=gradient_fiducial_rms,
        lmin=gradient_fit_ellmin,
        lmax=gradient_fit_ellmax,
    )



    # Deconvolve beam
    act_kmap = k150 / act_150_kbeam2d
    tclaa = tclaa_150 / (act_150_kbeam2d ** 2.0)
    act_kmap[~np.isfinite(act_kmap)] = 0
    tclaa[~np.isfinite(tclaa)] = 0

    plc_kmap = kp / plc_kbeam2d
    tclpp = tclpp / (plc_kbeam2d ** 2.0)
    plc_kmap[~np.isfinite(plc_kmap)] = 0
    tclpp[~np.isfinite(tclpp)] = 0

    # Fit cross-power of gradient and high-res; not usually used
    cents, c_ap = lbinner.bin(pow(act_kmap, plc_kmap) / w2)
    tclap = fit_p1d(
        l_edges, cents, c_ap, "apcross", modlmap, None, None, rms=0, lmin=highres_fit_ellmin, lmax=gradient_fit_ellmax
    )
    
    """ 
    !! LENS RECONSTRUCTION
    """

    # build symlens dictionary for lensing reconstruction
    feed_dict = {
        "uC_T_T": ucltt,  # goes in the lensing response func = lensed theory
        "tC_A_T_A_T": tclaa,  # the approximate ACT power spectrum, ACT beam deconvolved
        "tC_P_T_P_T": tclpp,  # approximate Planck power spectrum, Planck beam deconvolved
        "tC_A_T_P_T": tclap,  # same lensed theory as above, no instrumental noise
        "tC_P_T_A_T": tclap,  # same lensed theory as above, no instrumental noise
        "X": plc_kmap,  # gradient leg : 2D Planck map, Planck beam deconvolved
        "Y": act_kmap,  # hres leg : 2D ILC ACT map, ACT beam deconvolved 
    }

    # Sanity check
    for key in feed_dict.keys():
        assert np.all(np.isfinite(feed_dict[key]))
        
    # ask for reconstruction in Fourier space
    cqe = symlens.QE(
        shape,
        wcs,
        feed_dict,
        estimator="hdv_curl" if curl else "hdv", 
        XY="TT",
        xmask=xmask,
        ymask=ymask,
        field_names=["P", "A"],
        groups=None,
        kmask=kmask,
    )
    
    # Fourier space lens reconstruction
    krecon = cqe.reconstruct(feed_dict, xname="X_l1", yname="Y_l2", physical_units=True)

    # transform to real space for unweighted stack
    kappa = enmap.ifft(krecon, normalize="phys").real

    """ 
    !! REJECT WEIRD KAPPA
    """

    ##### temporary 3: to get rid of stamps with tSZ cluster in random locations
    if np.any(np.abs(kappa) > 15):
        continue
    
    # Unweighted stack 
    if i == 0:
        #kappa_final = np.zeros((len(ras), kappa.shape[0], kappa.shape[1]), dtype=np.float32)
        kappa_final = np.zeros((kappa.shape[0], kappa.shape[1]), dtype=np.float32)
    kappa_final += kappa[:, :]
    # TESTING!!!!!!!!!
    if i == 100000:
        break
np.save("/pscratch/sd/b/boryanah/ACTxDESI/ACT/kappa/kappa.npy", kappa_final)


