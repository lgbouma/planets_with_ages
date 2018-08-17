# -*- coding: utf-8 -*-
from __future__ import division, print_function
'''
DESCRIPTION
----------

We look at the CKS (paper 7) isochrone ages.

We also get the photometric rotation period measurements of CKS stars.
... and then calculate gyrochronological ages from them (& B-V colors).

... we also calculate activity-induced ages using R_HK' and Mamajek's
    calibrations, as another sanity check.

And we then make plots to see how the detected planet population changes.

What we actually care about is occurrence rate evolution in time. This requires
the Kepler stellar sample, and replicating e.g., Petigura's cuts in CKS IV, or
in CKS VII.

USAGE
----------

Select your desired plots from

    plot_wellmeasured = False
    plot_janky = False
    plot_boxplot = False
    plot_stacked_histograms = False
    plot_quartile_scatter = True
    plot_octile_scatter = True

then

$ python cks_prot_exploration.py

'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import os

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

from age_plots import plot_wellmeasuredparam, plot_jankyparam, \
    make_age_histograms_get_f_inds, make_stacked_histograms, \
    make_quartile_scatter, make_boxplot_koi_count, make_octile_scatter, \
    make_old_young_scatter

from download_furlan_2017_results import \
        download_furlan_radius_correction_table

def arr(x):
    return np.array(x)

if __name__ == '__main__':

    plot_wellmeasured = False
    plot_janky = False
    plot_boxplot = False
    plot_stacked_histograms = False
    plot_quartile_scatter = False
    plot_octile_scatter = False
    plot_metallicity_controlled = False
    plot_metallicity_controlled_pcttiles = True

    if plot_wellmeasured:
        print('plotting well measured params...')
    if plot_janky:
        print('plotting janky params...')

    # There are 1305 CKS spectra of KOIs with 2025 planet candidates. The
    # overlapping CKS stellar samples are: (1) magnitude-limited, Kp < 14.2,
    # (2) multi-planet hosts (3) USP hosts (4) HZ hosts (5) other.  Column data
    # are in ../data/cks-column-definitions.txt. These data are from the CKS
    # website, 2018/07/30, from paper II.
    df_II = pd.read_csv('../data/cks_physical_merged.csv')

    # These data are currently secret, from CKS paper VII.
    df_VII = pd.read_csv('../data/tab_star-machine.csv')

    sel = arr(make_age_histograms_get_f_inds(df_VII))
    df_VII['selected'] = sel

    # cks VII table only has star data. merge against cks II table, using
    # id_starname column, to get planet data too.
    df = pd.merge(df_VII, df_II, how='left', on='id_starname',
                  suffixes=('_VII', '_II'))
    sel = arr(df['selected'])

    # get Furlan+ 2017 table. Format in id_starname in same 0-padded foramt as
    # CKS.
    furlan_fname = '../data/Furlan_2017_table9.csv'
    if not os.path.exists(furlan_fname):
        download_furlan_radius_correction_table()
    f17 = pd.read_csv(furlan_fname)
    f17['id_starname'] = arr( ['K'+'{0:05d}'.format(koi) for koi in
                               arr(f17['KOI'])] )
    # sanity check that above works, i.e. that "id_starname" is just the KOI
    # number with strings padded.
    assert np.array_equal(arr(df['id_starname']),
                          arr([n[:-3] for n in arr(df['id_koicand'])]) )

    df = pd.merge(df, f17, how='left', on='id_starname', suffixes=('cks','f17'))

    ##########
    # all cuts from Petigura+ 2018 CKS IV
    sel = np.isfinite(arr(df['iso_prad']))
    sel &= arr(df['iso_prad']) < 32
    sel &= np.isfinite(arr(df['giso_slogage']))

    # apply all filters from table 1 of Petigura+ 2018
    # Kp<14.2
    sel &= np.isfinite(arr(df['kic_kepmag']))
    sel &= arr(df['kic_kepmag']) < 14.2
    # Teff=4700-6500K
    sel &= np.isfinite(arr(df['iso_steff']))
    sel &= arr(df['iso_steff']) < 6500
    sel &= arr(df['iso_steff']) > 4700
    # logg=3.9-5dex
    sel &= np.isfinite(arr(df['iso_slogg']))
    sel &= arr(df['iso_slogg']) < 5
    sel &= arr(df['iso_slogg']) > 3.9
    # P<350days
    sel &= np.isfinite(arr(df['koi_period']))
    sel &= arr(df['koi_period']) < 350
    # not a false positive
    is_fp = arr(df['cks_fp'])
    sel &= ~is_fp
    # not grazing (b<0.9)
    sel &= np.isfinite(arr(df['koi_impact']))
    sel &= arr(df['koi_impact']) < 0.9
    # radius correction factor <5%, logical or not given in Furlan+2017 table.
    sel &= ( (arr(df['Avg']) < 1.05) | (~np.isfinite(df['Avg']) ) )

    _ = arr(make_age_histograms_get_f_inds(df[sel]))

    # raw exploration plots
    goodparams = ['koi_period', 'iso_prad', 'koi_count','koi_dor']
    jankyparams = ['cks_smet_VII']

    for logy in [True, False]:
        for logx in [True, False]:
            for goodparam in goodparams:
                if not plot_wellmeasured:
                    continue
                plot_wellmeasuredparam(df, sel, goodparam, logx, logy,
                                       is_cks=True)

    for logy in [True, False]:
        for logx in [True, False]:
            for jankyparam in jankyparams:
                if not plot_janky:
                    continue
                plot_jankyparam(df, sel, jankyparam, logx, logy, is_cks=True)

    if plot_boxplot:
        print('making koi count boxplot')
        _make_boxplot_koi_count(df[sel])

    ######################
    # all CKS below here #
    ######################
    if plot_stacked_histograms:
        print('plotting stacked histograms')
        make_stacked_histograms(df[sel], logtime=True, xparam='aoverRstar')
        make_stacked_histograms(df[sel], logtime=False, xparam='aoverRstar')
        make_stacked_histograms(df[sel], logtime=True, xparam='period')
        make_stacked_histograms(df[sel], logtime=False, xparam='period')
        make_stacked_histograms(df[sel], logtime=False, xparam='radius')
        make_stacked_histograms(df[sel], logtime=True, xparam='radius')

    if plot_quartile_scatter:
        make_quartile_scatter(df[sel], xparam='koi_period')
        make_quartile_scatter(df[sel], xparam='koi_dor')

    if plot_octile_scatter:
        make_octile_scatter(df[sel], xparam='koi_period')
        make_octile_scatter(df[sel], xparam='koi_dor')

    # try to control for metallicity
    if plot_metallicity_controlled:

        make_old_young_scatter(df, xparam='koi_period', metlow=-0.05, methigh=0.05)

        binstr = 'binned_pm_pt05_FeH'
        metsel = arr(df['cks_smet_VII']) <= 0.05
        metsel &= arr(df['cks_smet_VII']) >= -0.05
        outpath = '../results/cks_age_plots/log_age_vs_cks_smet_VII_{:s}.pdf'.format(binstr)
        logx, logy = False, True
        plot_jankyparam(df, metsel, 'cks_smet_VII', logx, logy, is_cks=True,
                        outpdfpath=outpath, elw=0.2, ealpha=0.02)

        _ = make_age_histograms_get_f_inds(df[metsel],
                                           outstr='_smet_{:s}'.format(binstr),
                                           logage=True)

    if plot_metallicity_controlled_pcttiles:
        # Q: at fixed metallicity, does age affect the number of detected
        # planets in the Rp vs Porb plane?
        # A: if at all, difficult to see.

        mets = [(-0.25,-0.15),(-0.15,-0.05),(-0.05,0.05),(0.05,0.15),(0.15,0.25)]
        for met in mets:
            make_old_young_scatter(df, xparam='koi_period', metlow=met[0],
                                   methigh=met[1])
            make_old_young_scatter(df, xparam='koi_dor', metlow=met[0],
                                   methigh=met[1])

        # Q: at fixed age, does metallicity affect the number of detected
        # planets in the Rp vs Porb plane?
        logages = [(9.2,9.4),(9.4,9.6),(9.6,9.8),(9.8,10.0)]
        for logage in logages:
            make_old_young_scatter(df, xparam='koi_period', logagelow=logage[0],
                                   logagehigh=logage[1])
            #make_old_young_scatter(df, xparam='koi_dor', metlow=met[0],
            #                       methigh=met[1])

