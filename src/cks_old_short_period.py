# -*- coding: utf-8 -*-
'''
from `cks_age_exploration`, we found that there are not many old and close-in
planets. Where are all the short-period planets around the oldest stars?

----------
usage: select your filters (approach #1,#2,#3 as described in comments). then:

    `python cks_old_short_period.py`
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import os

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

from age_plots import plot_wellmeasuredparam, plot_jankyparam, \
    make_age_histograms_get_f_inds, make_stacked_histograms, \
    make_quartile_scatter, make_boxplot_koi_count, make_octile_scatter, \
    make_old_young_scatter, plot_scatter, plot_hr, \
    turnoff_v_mainsequence_scatters

from download_furlan_2017_results import \
    download_furlan_radius_correction_table

from cks_age_exploration import _get_cks_data, \
    _apply_cks_IV_metallicity_study_filters, \
    _apply_cks_IV_filters_plus_gaia_astrom_excess

from numpy import array as arr

from astropy import units as u, constants as const

def calculate_and_print_fractions(df, sel):
    '''
    do the rough calculation of "what %age of planets are at P<~2 or 3
    days >10 Gyr, and from 1-8gyr?"
    '''
    d = df[sel]
    agecuts = [9e9,10e9,11e9]
    periodcuts = [1,2,3]
    abyRcuts = [6,8,10]

    print('\n')
    for agecut in agecuts:
        for periodcut in periodcuts:
            old_subsel = (
                (10**(d['giso_slogage'])>agecut)
            )
            young_subsel = (
                (10**(d['giso_slogage'])<agecut)
            )
            old_pcut = (
                (10**(d['giso_slogage'])>agecut) & (d['koi_period']<periodcut)
            )
            young_pcut = (
                (10**(d['giso_slogage'])<agecut) & (d['koi_period']<periodcut)
            )

            print('For systems >{:.1E} yrs, {:.1f}% have P<{:.1f} d'.format(
                agecut,100*len(d[old_pcut])/len(d[old_subsel]), periodcut)
            )
            print('For systems <{:.1E} yrs, {:.1f}% have P<{:.1f} d'.format(
                agecut,100*len(d[young_pcut])/len(d[young_subsel]), periodcut)
            )

    print('\n')
    for agecut in agecuts:
        for abyRcut in abyRcuts:
            old_subsel = (
                (10**(d['giso_slogage'])>agecut)
            )
            young_subsel = (
                (10**(d['giso_slogage'])<agecut)
            )
            old_pcut = (
                (10**(d['giso_slogage'])>agecut) & (d['koi_period']<abyRcut)
            )
            young_pcut = (
                (10**(d['giso_slogage'])<agecut) & (d['koi_period']<abyRcut)
            )

            print('For systems >{:.1E} yrs, {:.1f}% have a/R<{:.1f}'.format(
                agecut,100*len(d[old_pcut])/len(d[old_subsel]), abyRcut)
            )
            print('For systems <{:.1E} yrs, {:.1f}% have a/R<{:.1f}'.format(
                agecut,100*len(d[young_pcut])/len(d[young_subsel]), abyRcut)
            )


def plot_ks2sample_abyRstar_old_v_young(df, agecut, savdir=None,
                                        abyRstar_str='koi_dor'):

    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))

    df_old = df[(10**(df['giso_slogage'])>=agecut)]
    df_young = df[(10**(df['giso_slogage'])<agecut)]

    counts, bin_edges = np.histogram(arr(df_old[abyRstar_str]),
                                     bins=len(df_old),
                                     normed=True)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf/cdf[-1],
            label="{:d} ``old'' (>{:.1e} yr)".format(len(df_old), agecut),
            lw=0.5)

    counts, bin_edges = np.histogram(arr(df_young[abyRstar_str]),
                                     bins=len(df_young),
                                     normed=True)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf/cdf[-1],
            label="{:d} ``young'' (<{:.1e} yr)".format(len(df_young), agecut),
            lw=0.5)

    ax.legend(loc='upper left', fontsize='xx-small')

    # tests for statistical signifiance
    from scipy import stats
    D, ks_p_value = stats.ks_2samp(arr(df_old[abyRstar_str]),
                                   arr(df_young[abyRstar_str]))
    _, _, ad_p_value = stats.anderson_ksamp([arr(df_old[abyRstar_str]),
                                            arr(df_young[abyRstar_str])])

    # check differences in Mp/Mstar btwn old and young population
    from cks_multis_vs_age import _get_WM14_mass
    old_masses = []
    for rp in arr(df_old['giso_prad']):
        old_masses.append(_get_WM14_mass(float(rp)))
    old_pmasses = arr(old_masses)*u.Mearth
    old_smasses = arr(df_old['giso_smass'])*u.Msun
    df_old['Mp_by_Mstar'] = (old_pmasses/old_smasses).cgs.value

    young_masses = []
    for rp in arr(df_young['giso_prad']):
        young_masses.append(_get_WM14_mass(float(rp)))
    young_masses = arr(young_masses)
    young_pmasses = arr(young_masses)*u.Mearth
    young_smasses = arr(df_young['giso_smass'])*u.Msun
    df_young['Mp_by_Mstar'] = (young_pmasses/young_smasses).cgs.value

    oldgiant = (df_old['giso_prad'] > 4)
    younggiant = (df_young['giso_prad'] > 4)

    # NOTE: if you don't see Mp/Mstar dependence (e.g., it's actually the LESS
    # MASSIVE ones disappearing)--> might be dynamically being throw ut
    txt = (
        'p={:.1e} for 2sampleKS old vs young'.format(ks_p_value)+
        '\np={:.1e} for 2sampleAD old vs young'.format(ad_p_value)+
        '\n<Mp/Mstar> old = {:.1e}, <Mp/Mstar> young = {:.1e} (incl Rp>4Rp)'.
        format(
            np.mean(df_old['Mp_by_Mstar']), np.mean(df_young['Mp_by_Mstar'])
        )+
        '\n<Mp/Mstar> old = {:.1e}, <Mp/Mstar> young = {:.1e} (not Rp>4Rp)'.
        format(
            np.mean(df_old[~oldgiant]['Mp_by_Mstar']),
            np.mean(df_young[~younggiant]['Mp_by_Mstar'])
        )+
        '\n<Rstar> old = {:.3e}, <Rstar> young = {:.3e}'.
        format(
            np.mean(df_old['giso_srad']),
            np.mean(df_young['giso_srad'])
        )
    )
    ax.text(0.95, 0.05, txt,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize='xx-small')

    if abyRstar_str=='koi_dor':
        xlabel='koi a/rstar'
    elif abyRstar_str=='cks_VII_dor':
        xlabel='CKS VII a/rstar'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('cdf')
    ax.set_xscale('log')

    # save it
    f.tight_layout()
    savpath = (savdir+
               "ks2sample_{:s}_old_v_young_cut{:.1e}yr.png".
               format(abyRstar_str, agecut)
              )
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))


def age_vs_abyRstar_classified_scatter(df_turnedoff, df_onMS, savdir=None,
                                       abyRstar_str='koi_dor'):

    plt.close('all')

    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
    ax.scatter(df_turnedoff[abyRstar_str], 10**(df_turnedoff['giso_slogage'])/1e9,
               marker='s', s=8, zorder=1, rasterized=True, label='turned off')
    ax.scatter(df_onMS[abyRstar_str], 10**(df_onMS['giso_slogage'])/1e9,
               marker='s', s=8, zorder=1, rasterized=True, label='on MS')

    ax.set_ylabel('age [Gyr] (cks VII gaia+CKS isochrone)')
    xscale='log'
    ax.set_xscale(xscale)
    if abyRstar_str=='koi_dor':
        ax.set_xlabel('koi a/Rstar')
    elif abyRstar_str=='cks_VII_dor':
        ax.set_xlabel('CKS-VII a/Rstar')

    ax.legend(loc='best',fontsize='small')

    f.tight_layout()

    savstr = '_classified_scatter'
    fname_pdf = 'age_vs_log_{:s}{:s}.pdf'.format(abyRstar_str, savstr)
    fname_png = fname_pdf.replace('.pdf','.png')
    f.savefig(savdir+fname_pdf)
    print('saved {:s}'.format(fname_pdf))
    f.savefig(savdir+fname_png, dpi=250)


def teffdist_vs_abyRstar_classified_scatter(df_turnedoff, df_onMS,
                                            teff_boundary_turnedoff,
                                            teff_boundary_onMS, savdir=None,
                                            show_onMS=True, logy=False,
                                            absy=False, abyRstar_str='koi_dor'):

    plt.close('all')

    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))

    if logy or absy:
        ax.scatter(
            df_turnedoff[abyRstar_str],
            np.abs(np.array(df_turnedoff['cks_steff'])-teff_boundary_turnedoff),
            marker='s', s=8, zorder=1, rasterized=True, label='turned off'
        )
    else:
        ax.scatter(
            df_turnedoff[abyRstar_str],
            np.array(df_turnedoff['cks_steff'])-teff_boundary_turnedoff,
            marker='s', s=8, zorder=1, rasterized=True, label='turned off'
        )

    if show_onMS:
        ax.scatter(
            df_onMS[abyRstar_str],
            np.array(df_onMS['cks_steff'])-teff_boundary_onMS, marker='s', s=8,
            zorder=1, rasterized=True, label='on MS'
        )

    if not logy and not absy:
        ax.set_ylabel(
            '$ T_{\mathrm{eff,\ actual}}$ - $T_{\mathrm{eff,\ dividing\ line}} $ [K]'
        )
    else:
        ax.set_ylabel(
            '$| T_{\mathrm{eff,\ actual}}$ - $T_{\mathrm{eff,\ dividing\ line}}| $ [K]'
        )
    xscale='log'
    ax.set_xscale(xscale)
    if abyRstar_str=='koi_dor':
        ax.set_xlabel('koi a/Rstar')
    elif abyRstar_str=='cks_VII_dor':
        ax.set_xlabel('CKS-VII a/Rstar')

    ax.legend(loc='best',fontsize='small')

    if logy:
        ax.set_yscale('log')

    f.tight_layout()

    savstr = (
        '_classified_scatter' if show_onMS
        else '_classified_scatter_onlyturnedoff'
    )
    logstr = 'log' if logy else ''
    absstr = 'abs' if absy else ''
    fname_pdf = ('{:s}{:s}teffdist_vs_log_{:s}{:s}.pdf'.
                 format(logstr,absstr,abyRstar_str,savstr))
    fname_png = fname_pdf.replace('.pdf','.png')
    f.savefig(savdir+fname_pdf)
    print('saved {:s}'.format(fname_pdf))
    f.savefig(savdir+fname_png, dpi=250)


def logg_vs_teff_classified_scatter(df_turnedoff, df_onMS, savdir=None):

    savstr = '_classified_scatter'
    plt.close('all')

    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
    ax.scatter(df_turnedoff['cks_steff'], df_turnedoff['giso_slogg'],
               marker='s', s=8, zorder=1, rasterized=True, label='turned off')
    ax.scatter(df_onMS['cks_steff'], df_onMS['giso_slogg'], marker='s', s=8,
               zorder=1, rasterized=True, label='on MS')

    df = pd.read_csv('../data/logg_vs_teff_line.csv',
                     names=['teff','logg'], header=None)
    logg = np.array(df['logg'])
    teff = np.array(df['teff'])

    from scipy.interpolate import interp1d
    fn = interp1d(teff[::-1], logg[::-1], kind='quadratic', bounds_error=True)

    teff_arr = np.linspace(np.min(teff),np.max(teff),1000)
    logg_arr = fn(teff_arr)

    ax.plot(teff_arr, logg_arr, zorder=2)

    ax.set_ylabel('logg (cks VII gaia+CKS isochrone)')
    xscale='linear'
    ax.set_xscale(xscale)
    ax.set_xlabel('cks teff [K]')

    ax.legend(loc='best',fontsize='small')

    xlim = ax.get_xlim()
    ax.set_xlim(max(xlim),min(xlim))
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim),min(ylim))

    f.tight_layout()

    fname_pdf = 'logg_vs_{:s}teff{:s}.pdf'.format(xscale, savstr)
    fname_png = fname_pdf.replace('.pdf','.png')
    f.savefig(savdir+fname_pdf)
    print('saved {:s}'.format(fname_pdf))
    f.savefig(savdir+fname_png, dpi=250)


def plot_ks2sample_abyRstar_turnoff_v_mainsequence(df, savdir=None,
                                                   abyRstar_str='koi_dor'):

    # get the stars past the "turnoff", and those not.
    _df = pd.read_csv('../data/logg_vs_teff_line.csv',
                     names=['teff','logg'], header=None)
    _logg = np.array(_df['logg'])
    _teff = np.array(_df['teff'])

    from scipy.interpolate import interp1d
    fn = interp1d(_teff[::-1], _logg[::-1], kind='quadratic',
                  bounds_error=False, fill_value='extrapolate')
    fn2 = interp1d(_logg[::-1], _teff[::-1], kind='quadratic',
                   bounds_error=False, fill_value='extrapolate')

    teff_arr = np.linspace(np.min(_teff),np.max(_teff),1000)
    logg_arr = fn(teff_arr)

    slogg = arr(df['giso_slogg'])
    steff = arr(df['cks_steff'])

    sel = (steff > 5000) & (slogg <= np.max(_logg))
    subsel = ( slogg < fn(steff) )
    subsel &= ( steff < fn2(slogg) )

    df_turnedoff = df[sel & subsel]
    df_onMS = df[sel & ~subsel]

    teff_boundary_turnedoff = fn2(np.array(df_turnedoff['giso_slogg']))
    teff_boundary_onMS = fn2(np.array(df_onMS['giso_slogg']))

    # make classified scatter plot
    logg_vs_teff_classified_scatter(df_turnedoff, df_onMS, savdir=savdir)
    age_vs_abyRstar_classified_scatter(df_turnedoff, df_onMS, savdir=savdir,
                                       abyRstar_str=abyRstar_str)
    teffdist_vs_abyRstar_classified_scatter(df_turnedoff, df_onMS,
                                            teff_boundary_turnedoff,
                                            teff_boundary_onMS, savdir=savdir,
                                            abyRstar_str=abyRstar_str)
    teffdist_vs_abyRstar_classified_scatter(df_turnedoff, df_onMS,
                                            teff_boundary_turnedoff,
                                            teff_boundary_onMS, savdir=savdir,
                                            show_onMS=False, logy=False,
                                            absy=True,
                                            abyRstar_str=abyRstar_str)
    teffdist_vs_abyRstar_classified_scatter(df_turnedoff, df_onMS,
                                            teff_boundary_turnedoff,
                                            teff_boundary_onMS, savdir=savdir,
                                            show_onMS=False, logy=True,
                                            abyRstar_str=abyRstar_str)

    # make the cdf plot
    plt.close('all')
    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))

    counts, bin_edges = np.histogram(arr(df_turnedoff[abyRstar_str]),
                                     bins=len(df_turnedoff),
                                     normed=True)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf/cdf[-1],
            label="{:d} ``turnedoff''".format(len(df_turnedoff)),
            lw=0.5)

    counts, bin_edges = np.histogram(arr(df_onMS[abyRstar_str]),
                                     bins=len(df_onMS),
                                     normed=True)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf/cdf[-1],
            label="{:d} ``on MS''".format(len(df_onMS)),
            lw=0.5)

    ax.legend(loc='upper left', fontsize='xx-small')

    # tests for statistical signifiance
    from scipy import stats
    D, ks_p_value = stats.ks_2samp(arr(df_turnedoff[abyRstar_str]),
                                   arr(df_onMS[abyRstar_str]))
    _, _, ad_p_value = stats.anderson_ksamp([arr(df_turnedoff[abyRstar_str]),
                                            arr(df_onMS[abyRstar_str])])

    # check differences in Mp/Mstar btwn old and young population
    from cks_multis_vs_age import _get_WM14_mass
    turnedoff_masses = []
    for rp in arr(df_turnedoff['giso_prad']):
        turnedoff_masses.append(_get_WM14_mass(float(rp)))
    turnedoff_pmasses = arr(turnedoff_masses)*u.Mearth
    turnedoff_smasses = arr(df_turnedoff['giso_smass'])*u.Msun
    df_turnedoff['Mp_by_Mstar'] = (turnedoff_pmasses/turnedoff_smasses).cgs.value

    mainseq_masses = []
    for rp in arr(df_onMS['giso_prad']):
        mainseq_masses.append(_get_WM14_mass(float(rp)))
    mainseq_masses = arr(mainseq_masses)
    mainseq_pmasses = arr(mainseq_masses)*u.Mearth
    mainseq_smasses = arr(df_onMS['giso_smass'])*u.Msun
    df_onMS['Mp_by_Mstar'] = (mainseq_pmasses/mainseq_smasses).cgs.value

    turnedoffgiant = (df_turnedoff['giso_prad'] > 4)
    mainseqgiant = (df_onMS['giso_prad'] > 4)

    # NOTE: if you don't see Mp/Mstar dependence (e.g., it's actually the LESS
    # MASSIVE ones disappearing)--> might be dynamically being throw ut
    txt = (
        'p={:.1e} for 2sampleKS turnoff vs MS'.format(ks_p_value)+
        '\np={:.1e} for 2sampleAD turnoff vs MS'.format(ad_p_value)+
        '\n<Mp/Mstar> turnoff = {:.1e}, <Mp/Mstar> MS = {:.1e} (incl Rp>4Rp)'.
        format(
            np.mean(df_turnedoff['Mp_by_Mstar']), np.mean(df_onMS['Mp_by_Mstar'])
        )+
        '\n<Mp/Mstar> turnoff = {:.1e}, <Mp/Mstar> MS = {:.1e} (not Rp>4Rp)'.
        format(
            np.mean(df_turnedoff[~turnedoffgiant]['Mp_by_Mstar']),
            np.mean(df_onMS[~mainseqgiant]['Mp_by_Mstar'])
        )+
        '\n<Rstar> turnoff {:.2e}, <Rstar> MS = {:.2e}'.
        format(
            np.mean(df_turnedoff['giso_srad']),
            np.mean(df_onMS['giso_srad'])
        )
    )

    ax.text(0.95, 0.05, txt,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize='xx-small')

    if abyRstar_str=='koi_dor':
        ax.set_xlabel('koi a/Rstar')
    elif abyRstar_str=='cks_VII_dor':
        ax.set_xlabel('CKS-VII a/Rstar')
    ax.set_ylabel('cdf')
    ax.set_xscale('log')

    f.tight_layout()
    savpath = ( savdir +
               "ks2sample_{:s}_turnoff_v_mainsequence.png".format(abyRstar_str)
              )
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))
    savpath = savpath.replace('.png','.pdf')
    f.savefig(savpath, bbox_inches="tight")
    print("made {:s}".format(savpath))

    return df_turnedoff, df_onMS


def _take_innermost_filter(df):
    # impose a/Rstar < 100
    # only take the innermost detected transiting planet of any given system

    # construct column to select innermost objects of systems
    is_innermost = []
    for ix,row in df.iterrows():
        sname = row['id_starname']
        this_sys = df[ df['id_starname'] == sname ]
        if float(row['koi_dor']) == np.min( this_sys['koi_dor'] ):
            is_innermost.append(True)
        else:
            is_innermost.append(False)
    is_innermost = np.array(is_innermost)

    sel = np.array(df['cks_VII_dor'] < 100)

    return is_innermost & sel


def make_old_short_period_plots():

    # approach #1: just use Petigura+ 2018's filters
    # approach #2: just use Petigura+ 2018's filters, + filter on gaia
    # astrometric excess
    # approach #3: just use Petigura+ 2018's filters, + filter on gaia
    # astrometric excess, + a/Rstar<100 only, + only the innermost planets of
    # systems.
    do_approach1 = 0
    do_approach2 = 0
    do_approach3 = 1
    approaches = np.array(
        [do_approach1,do_approach2,do_approach3]).astype(bool)
    if not len(approaches[approaches])==1:
        raise AssertionError('only one approach allowed')

    make_all = 0
    if make_all:
        make_initial_plots = 1
        make_stacked_histogram_plots = 1
        make_sanity_check_scatters = 1
        make_hr_diagram = 1
        make_ks2sample_abyRstar = 1
        make_ks2sample_turnoff = 1
        make_turnoff_v_mainsequence_scatters = 1
    else:
        make_initial_plots = 1
        make_stacked_histogram_plots = 0
        make_sanity_check_scatters = 0
        make_hr_diagram = 0
        make_ks2sample_abyRstar = 0
        make_ks2sample_turnoff = 0
        make_turnoff_v_mainsequence_scatters = 0


    if do_approach1:
        df = _get_cks_data()
        sel = _apply_cks_IV_metallicity_study_filters(df)
        savdir = '../results/cks_age_plots_old_short_period_p18filters/'
        savdir_append='_old_short_period_p18filters'

    if do_approach2:
        df = _get_cks_data(merge_vs_gaia=True)
        sel = _apply_cks_IV_filters_plus_gaia_astrom_excess(df)
        savdir = '../results/cks_age_plots_old_short_period/'
        savdir_append='_old_short_period'

    if do_approach3:
        df = _get_cks_data(merge_vs_gaia=True)
        sel = _apply_cks_IV_filters_plus_gaia_astrom_excess(df)
        sel &= _take_innermost_filter(df)
        savdir = '../results/cks_age_plots_old_short_period_onlyinnermost/'
        savdir_append='_old_short_period_onlyinnermost'

    # remake the plots that got us interested in this
    if make_initial_plots:
        for xparam in ['koi_period', 'koi_dor', 'cks_VII_dor']:
            for logy in [True,False]:
                for logx in [True,False]:
                    plot_wellmeasuredparam(df, sel, xparam, logx, logy,
                                           is_cks=True,
                                           savdir_append=savdir_append)
                    plot_wellmeasuredparam(df, sel, xparam, logx, logy,
                                           is_cks=True,
                                           savdir_append=savdir_append)

    if make_stacked_histogram_plots:
        make_stacked_histograms(df[sel], logtime=False, xparam='koi_dor',
                                savdir=savdir)
        make_stacked_histograms(df[sel], logtime=False, xparam='cks_VII_dor',
                                savdir=savdir)
        make_stacked_histograms(df[sel], logtime=False, xparam='period',
                                savdir=savdir)
        make_stacked_histograms(df[sel], logtime=True, xparam='koi_dor',
                                savdir=savdir)
        make_stacked_histograms(df[sel], logtime=True, xparam='cks_VII_dor',
                                savdir=savdir)
        make_stacked_histograms(df[sel], logtime=True, xparam='period',
                                savdir=savdir)
        make_quartile_scatter(df[sel], xparam='koi_period',
                              savdir=savdir)
        make_quartile_scatter(df[sel], xparam='koi_dor',
                              savdir=savdir)
        make_quartile_scatter(df[sel], xparam='cks_VII_dor',
                              savdir=savdir)

    if make_sanity_check_scatters:
        # a few sanity checks
        plot_scatter(df, sel, 'koi_period', 'cks_smet', True, False, is_cks=True,
                     savdir=savdir, ylim=[0.5,-0.5])
        plot_scatter(df, sel, 'koi_dor', 'cks_smet', True, False, is_cks=True,
                     savdir=savdir, ylim=[0.5,-0.5])
        plot_scatter(df, sel, 'cks_VII_dor', 'cks_smet', True, False, is_cks=True,
                     savdir=savdir, ylim=[0.5,-0.5])
        plot_scatter(df, sel, 'cks_smet', 'giso_slogage', False, False,
                     is_cks=True, savdir=savdir, xlim=[0.5,-0.5])

    # do the rough calculation of "what %age of planets are at a/R<10 at
    # >10 Gyr, and below it?
    calculate_and_print_fractions(df, sel)

    # what is the a/Rstar<3, smet < -0.3 object?
    inds = (df['cks_smet'] < -0.3)
    inds &= (df['koi_dor'] < 3)
    print(df[inds])

    # what are the a/Rstar<3, age > 10Gyr objects?
    inds = (10**(df['giso_slogage']) > 12e9)
    inds &= (df['koi_dor'] < 4)
    print(df[inds])

    # what exactly is the difference between a 14 gyro and 10 gyro isochrone
    # age?
    if make_hr_diagram:
        plot_hr(df, sel, 'giso_slogage', is_cks=True,
                savdir=savdir, yaxis='abskmag', xscale='log')
        plot_hr(df, sel, 'giso_slogage', is_cks=True,
                savdir=savdir, yaxis='abskmag', xscale='linear')
        plot_hr(df, sel, 'giso_slogage', is_cks=True,
                savdir=savdir, yaxis='logg', xscale='log')
        plot_hr(df, sel, 'giso_slogage', is_cks=True,
                savdir=savdir, yaxis='logg', xscale='linear')
        plot_hr(df, sel, 'giso_slogage', is_cks=True,
                savdir=savdir, yaxis='logg', savstr='turnoffcut',
                xscale='linear')

    # CDFs and two-sample KS to compare the a/Rstar distributions of old and
    # young planets. What is the p-value?
    if make_ks2sample_abyRstar:
        agecuts = [7e9,8e9,9e9,10e9,11e9,12e9]
        for agecut in agecuts:
            plot_ks2sample_abyRstar_old_v_young(df[sel], agecut, savdir=savdir,
                                               abyRstar_str='koi_dor')
            plot_ks2sample_abyRstar_old_v_young(df[sel], agecut, savdir=savdir,
                                               abyRstar_str='cks_VII_dor')

    if make_ks2sample_turnoff:
        _, _ = (
        plot_ks2sample_abyRstar_turnoff_v_mainsequence(df[sel], savdir=savdir,
                                                      abyRstar_str='koi_dor')
        )

        df_turnedoff, df_onMS = (
        plot_ks2sample_abyRstar_turnoff_v_mainsequence(df[sel], savdir=savdir,
                                                      abyRstar_str='cks_VII_dor')
        )

    if make_turnoff_v_mainsequence_scatters and make_ks2sample_turnoff:

        turnoff_v_mainsequence_scatters(df_turnedoff, df_onMS, savdir=savdir,
                                        xparam='koi_period')
        turnoff_v_mainsequence_scatters(df_turnedoff, df_onMS, savdir=savdir,
                                        xparam='koi_dor')
        turnoff_v_mainsequence_scatters(df_turnedoff, df_onMS, savdir=savdir,
                                        xparam='cks_VII_dor')


if __name__ == '__main__':
    make_old_short_period_plots()
