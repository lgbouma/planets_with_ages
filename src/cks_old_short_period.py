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
    make_old_young_scatter, plot_scatter

from download_furlan_2017_results import \
    download_furlan_radius_correction_table

from cks_age_exploration import _get_cks_data, \
    _apply_cks_IV_metallicity_study_filters

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


def make_old_short_period_plots():

    make_initial_plots = False

    df = _get_cks_data()
    sel = _apply_cks_IV_metallicity_study_filters(df)

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


if __name__ == '__main__':
    make_old_short_period_plots()
