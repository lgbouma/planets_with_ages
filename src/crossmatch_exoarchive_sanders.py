# -*- coding: utf-8 -*-
'''
DESCRIPTION
----------
Download exoplanet archive. Crossmatch the planets with Sanders and Das, 2018.
Make plots.

USAGE
----------

$ python crossmatch_exoarchive_sanders.py

Outputs a file, `../data/ea_sd18_matches.csv` with all the data you should
need.
'''

from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
import h5py

import os

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

def arr(x):
    return np.array(x)

# Sanders and Das 2018 catalog of stellar isochrone ages using Gaia
# information, combining with heterogeneous mix of RAVE, GALAH, APOGEE, LAMOST,
# and SEGUE surveys.  Most interesting columns:
# l, b, log10_age, log10_age_err, log10age_Z_corr, survey (e.g., APOGEE,
# LAMOST, GALAH ...), flag (0 if good, 1 if bad isochrones, 2 if bad spectra, 3
# if bad photometry, 4 if bad astrometry, 5 if bad mass, 6 if bad
# uncertainties, 7 if probable binaries).

# Note that their RAs and DECs are complete crap. This does not inspire me with
# confidence about the quality of their work.

# sort of jankily, only give symmetric log10_age_errs...

if __name__ == '__main__':

    sd18dir = '/home/luke/local/planets_with_ages/'
    sd18file= 'Sanders_Das_2018_gaia_spectro.hdf5'

    with h5py.File(sd18dir+sd18file, 'r') as f:

        print('reading the big table')
        sd18 = f['data']
        sd18 = sd18['l', 'b', 'log10_age', 'log10_age_err', 'log10age_Z_corr',
                    'survey', 'flag', 'mu_l', 'mu_b']
        print('done reading the big table')

    sd18_glon = arr(sd18['l'])*u.rad
    sd18_glat = arr(sd18['b'])*u.rad

    f_glon = np.isfinite(sd18_glon)
    f_glat = np.isfinite(sd18_glat)
    goodflag = (sd18['flag'] == 0)

    good_sd18_inds = (f_glon) & (f_glat) #& (goodflag)

    c_sd18 = SkyCoord(sd18_glon[good_sd18_inds],
                      sd18_glat[good_sd18_inds],
                      frame='galactic')

    c_sd18 = c_sd18.icrs

    # crossmatch vs exoarchive by ra and dec.
    ea_tab = NasaExoplanetArchive.get_confirmed_planets_table(
                                        all_columns=True, show_progress=True)

    ea_rad = arr(ea_tab['ra'])*u.deg
    ea_dec = arr(ea_tab['dec'])*u.deg

    c_ea = SkyCoord(ra=ea_rad, dec=ea_dec)

    seplimit = 10*u.arcsec

    # do the crossmatch. the first extra step helps by cacheing a kdtree.
    c_ea_sub = c_ea[:10]
    idx_ea_sub, idx_sd18, d2d, _ = c_sd18.search_around_sky(c_ea_sub, seplimit)

    print('beginning crossmatch')
    idx_ea, idx_sd18, d2d, _ = c_sd18.search_around_sky(c_ea, seplimit)
    print('completed crossmatch')

    # check your contamination fraction vs separation
    plt.close('all')
    plt.hist(d2d.to(u.arcsec), histtype='step', bins=20)
    plt.xlabel('2d distance (arcsec)')
    plt.ylabel('count')
    plt.savefig('../results/distance_check_hist.pdf')
    # result: 10 arcsecond separation is OK.

    ##########

    # select specific exoarchive columns to make life easier
    ea_matches = ea_tab[idx_ea]
    ea_matches.remove_column('sky_coord')

    ea_cols = ['pl_radj', 'pl_rade', 'pl_orbper',
               'pl_dens', 'pl_orbeccen', 'pl_orbincl',
               'st_age', 'st_dist', 'st_mass', 'st_teff']

    extra_cols = []
    for ea_col in ea_cols:
        plus_err = ea_col+'err1'
        minus_err = ea_col+'err2'
        extra_cols.append(plus_err)
        extra_cols.append(minus_err)

    ea_cols += extra_cols
    ea_cols += ['ra', 'dec',  'pl_discmethod', 'pl_name','pl_hostname','pl_pnum', 'st_glat']

    ea_matches_cut = ea_matches[ea_cols]

    # get rid of "mixin" columns in order to join exoarchive and SD18 matches
    _ = Table(np.array(ea_matches_cut))
    _['index'] = np.array(range(len(_)))

    sd18_match = Table(sd18[idx_sd18])
    sd18_match['ra'] = c_sd18[idx_sd18].ra.value
    sd18_match['dec'] = c_sd18[idx_sd18].dec.value
    sd18_match['index'] = np.array(range(len(sd18_match)))

    from astropy.table import join

    ea_sd18_matches = join(_, sd18_match, keys='index', join_type='outer',
                           table_names=['ea','sd18'])

    df = ea_sd18_matches.to_pandas()
    df.to_csv('../data/ea_sd18_matches.csv', index=False)
