'''
from `cks_age_exploration`, we found that there are not many old and close-in
planets. Where are all the short-period planets around the oldest stars?

let's look at this closer.
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
    make_old_young_scatter, plot_scatter, plot_hr

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


def plot_ks2sample_abyRstar_old_v_young(df, agecut):

    f, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))

    df_old = df[(10**(df['giso_slogage'])>=agecut)]
    df_young = df[(10**(df['giso_slogage'])<agecut)]

    counts, bin_edges = np.histogram(arr(df_old['koi_dor']),
                                     bins=len(df_old),
                                     normed=True)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf/cdf[-1],
            label="{:d} ``old'' (>{:.1e} yr)".format(len(df_old), agecut),
            lw=0.5)

    counts, bin_edges = np.histogram(arr(df_young['koi_dor']),
                                     bins=len(df_young),
                                     normed=True)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf/cdf[-1],
            label="{:d} ``young'' (<{:.1e} yr)".format(len(df_young), agecut),
            lw=0.5)

    ax.legend(loc='upper left', fontsize='xx-small')

    # tests for statistical signifiance
    from scipy import stats
    D, ks_p_value = stats.ks_2samp(arr(df_old['koi_dor']),
                                   arr(df_young['koi_dor']))
    _, _, ad_p_value = stats.anderson_ksamp([arr(df_old['koi_dor']),
                                            arr(df_young['koi_dor'])])

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
        )
    )
    ax.text(0.95, 0.05, txt,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize='xx-small')

    ax.set_xlabel('koi a/Rstar')
    ax.set_ylabel('cdf')
    ax.set_xscale('log')

    # save it
    f.tight_layout()
    savpath = ("../results/cks_age_plots_old_short_period/"
               "ks2sample_abyRstar_old_v_young_cut{:.1e}yr.png".format(agecut))
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))




def make_old_short_period_plots():

    make_initial_plots = False
    make_sanity_check_scatters = False
    make_hr_diagram = False
    make_ks2sample_abyRstar = True

    # df = _get_cks_data()
    # sel = _apply_cks_IV_metallicity_study_filters(df) #FIXME

    df = _get_cks_data(merge_vs_gaia=True)
    sel = _apply_cks_IV_filters_plus_gaia_astrom_excess(df)

    # remake the plots that got us interested in this
    if make_initial_plots:
        for xparam in ['koi_period', 'koi_dor']:
            logx, logy = True, False
            plot_wellmeasuredparam(df, sel, xparam, logx, logy,
                                   is_cks=True, savdir_append='_old_short_period')
            plot_wellmeasuredparam(df, sel, xparam, logx, logy,
                                   is_cks=True, savdir_append='_old_short_period')

        make_stacked_histograms(df[sel], logtime=False, xparam='aoverRstar',
                                savdir='../results/cks_age_plots_old_short_period/')
        make_stacked_histograms(df[sel], logtime=False, xparam='period',
                                savdir='../results/cks_age_plots_old_short_period/')
        make_stacked_histograms(df[sel], logtime=True, xparam='aoverRstar',
                                savdir='../results/cks_age_plots_old_short_period/')
        make_stacked_histograms(df[sel], logtime=True, xparam='period',
                                savdir='../results/cks_age_plots_old_short_period/')
        make_quartile_scatter(df[sel], xparam='koi_period',
                              savdir='../results/cks_age_plots_old_short_period/')
        make_quartile_scatter(df[sel], xparam='koi_dor',
                              savdir='../results/cks_age_plots_old_short_period/')

    if make_sanity_check_scatters:
        # a few sanity checks
        plot_scatter(df, sel, 'koi_period', 'cks_smet', True, False, is_cks=True,
                     savdir='../results/cks_age_plots_old_short_period/',
                     ylim=[0.5,-0.5])
        plot_scatter(df, sel, 'koi_dor', 'cks_smet', True, False, is_cks=True,
                     savdir='../results/cks_age_plots_old_short_period/',
                     ylim=[0.5,-0.5])
        plot_scatter(df, sel, 'cks_smet', 'giso_slogage', False, False,
                     is_cks=True,
                     savdir='../results/cks_age_plots_old_short_period/',
                     xlim=[0.5,-0.5])

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
                savdir='../results/cks_age_plots_old_short_period/')

    # CDFs and two-sample KS to compare the a/Rstar distributions of old and
    # young planets. What is the p-value?
    if make_ks2sample_abyRstar:
        agecuts = [7e9,8e9,9e9,10e9,11e9,12e9]
        for agecut in agecuts:
            plot_ks2sample_abyRstar_old_v_young(df[sel], agecut)


if __name__ == '__main__':
    make_old_short_period_plots()
