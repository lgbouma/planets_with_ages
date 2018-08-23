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

$ python cks_age_exploration.py

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
import astropy.constants as c

from age_plots import plot_wellmeasuredparam, plot_jankyparam, \
    make_age_histograms_get_f_inds, make_stacked_histograms, \
    make_quartile_scatter, make_boxplot_koi_count, make_octile_scatter, \
    make_old_young_scatter, plot_scatter

from download_furlan_2017_results import \
        download_furlan_radius_correction_table

def arr(x):
    return np.array(x)


def _get_cks_data():
    '''
    Returns:

        dataframe with CKS II planet data, supplemented by CKS VII stellar
        data, supplemented by Furlan+2017 dilution column.

    Description:

        There are 1305 CKS spectra of KOIs with 2025 planet candidates. The
        overlapping CKS stellar samples are: (1) magnitude-limited, Kp < 14.2,
        (2) multi-planet hosts (3) USP hosts (4) HZ hosts (5) other.  Column
        data are in ../data/cks-column-definitions.txt.

        df_II data are from the CKS website, 2018/07/30, and were released in
        paper II.

        df_VII data are currently (2018/08/04) secret, from CKS paper VII.

        since cks VII table only has star data, I merge it against cks II
        table, using id_starname column, to get planet data too.

        I then also merge against Furlan+2017 to get the dilution column,
        "Avg".
    '''

    df_VII_planets = pd.read_csv(
        '../data/Fulton_2018_CKS_VII_planet_properties.csv')

    df_VII_stars = pd.read_csv(
        '../data/Fulton_2018_CKS_VII_star_properties.csv')

    df_VII_planets['id_starname'] = np.array(
        df_VII_planets['id_koicand'].str.slice(start=0,stop=6)
    )

    df_II = pd.read_csv('../data/cks_physical_merged.csv')

    _ = pd.merge(df_VII_planets, df_VII_stars, how='left', on='id_starname',
                  suffixes=('_VII_p', '_VII_s'))

    # pull non-changed stellar parameters from CKS II
    subcols = ['kic_kepmag', 'id_starname', 'cks_fp', 'koi_impact',
               'koi_count', 'koi_dor', 'koi_model_snr']

    # avoid duplication, otherwise you get mixed merge
    df = pd.merge(_, df_II[subcols].drop_duplicates(subset=['id_starname']),
                  how='left', on='id_starname')

    # logg is not included, so we must calculate it
    M = np.array(df['giso_smass'])*u.Msun
    R = np.array(df['giso_srad'])*u.Rsun
    g = (c.G * M / R**2).cgs
    df['giso_slogg'] = np.array(np.log10(g.cgs.value))

    sel = arr(make_age_histograms_get_f_inds(df, actually_make_plots=False))

    df['selected'] = sel

    # nb. no need to match against Furlan17; 'fur17_rcorr_avg' has the relevant
    # column already.

    ### # get Furlan+ 2017 table. Format in id_starname in same 0-padded foramt
    ### # as CKS.
    ### furlan_fname = '../data/Furlan_2017_table9.csv'
    ### if not os.path.exists(furlan_fname):
    ###     download_furlan_radius_correction_table()
    ### f17 = pd.read_csv(furlan_fname)
    ### f17['id_starname'] = arr( ['K'+'{0:05d}'.format(koi) for koi in
    ###                            arr(f17['KOI'])] )
    ### # sanity check that above works, i.e. that "id_starname" is just the KOI
    ### # number with strings padded.
    ### assert np.array_equal(arr(df['id_starname']),
    ###                       arr([n[:-3] for n in arr(df['id_koicand'])]) )

    ### df = pd.merge(df, f17, how='left', on='id_starname', suffixes=('cks','f17'))

    return df


def _apply_cks_IV_metallicity_study_filters(df):
    '''
    given df from _get_cks_data, return the boolean indices for the
    subset of interest.

    (See Petigura+ 2018, CKS IV, table 1)
    '''

    sel = np.isfinite(arr(df['giso_prad']))
    sel &= arr(df['giso_prad']) < 32
    sel &= np.isfinite(arr(df['giso_slogage']))

    # apply all filters from table 1 of Petigura+ 2018
    # Kp<14.2
    sel &= np.isfinite(arr(df['kic_kepmag']))
    sel &= arr(df['kic_kepmag']) < 14.2
    # Teff=4700-6500K
    sel &= np.isfinite(arr(df['cks_steff']))
    sel &= arr(df['cks_steff']) < 6500
    sel &= arr(df['cks_steff']) > 4700
    # logg=3.9-5dex
    sel &= np.isfinite(arr(df['giso_slogg']))
    sel &= arr(df['giso_slogg']) < 5
    sel &= arr(df['giso_slogg']) > 3.9
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
    sel &= ( (arr(df['fur17_rcorr_avg']) < 1.05) | (~np.isfinite(df['fur17_rcorr_avg']) ) )

    return sel


if __name__ == '__main__':

    plot_wellmeasured = False
    plot_janky = False
    plot_boxplot = False
    plot_stacked_histograms = False
    plot_quartile_scatter = False
    plot_octile_scatter = False
    plot_metallicity_controlled = False
    plot_metallicity_controlled_pcttiles = False

    if plot_wellmeasured:
        print('plotting well measured params...')
    if plot_janky:
        print('plotting janky params...')

    df = _get_cks_data()
    sel = _apply_cks_IV_metallicity_study_filters(df)

    # raw exploration plots
    _ = arr(make_age_histograms_get_f_inds(df[sel]))

    goodparams = ['koi_period', 'giso_prad', 'koi_count','koi_dor']
    jankyparams = ['cks_smet']

    for logy in [True, False]:
        for logx in [True, False]:
            for xparam in goodparams:
                if not plot_wellmeasured:
                    continue
                plot_wellmeasuredparam(df, sel, xparam, logx, logy,
                                       is_cks=True)

    for logy in [True, False]:
        for logx in [True, False]:
            for xparam in jankyparams:
                if not plot_janky:
                    continue
                plot_jankyparam(df, sel, xparam, logx, logy, is_cks=True)

    if plot_boxplot:
        print('making koi count boxplot')
        make_boxplot_koi_count(df[sel])

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
        metsel = arr(df['cks_smet']) <= 0.05
        metsel &= arr(df['cks_smet']) >= -0.05
        outpath = '../results/cks_age_plots/log_age_vs_cks_smet_VII_{:s}.pdf'.format(binstr)
        logx, logy = False, True
        plot_jankyparam(df, metsel, 'cks_smet', logx, logy, is_cks=True,
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


    plot_scatter(df, sel, 'giso_prad', 'cks_smet', True, False, is_cks=True)

