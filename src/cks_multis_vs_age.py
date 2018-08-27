# -*- coding: utf-8 -*-
from __future__ import division, print_function
'''
DESCRIPTION
----------

1. Does the average separation between planet pairs in multis (in units of mutual
hill radii) increase for the oldest multis?

--> Maybe a bit for closest pairs (<30R_H separated), but not statistically
    significant

2. Does the innermost planet of a multi tend to be detected further away from the
host star in the oldest systems?

--> well, the median a/Rstar in the youngest quartile is 12.9
    the median a/Rstar in the oldest quartile is 14.1.
    it does not look like a clean story though.

3. How does the average number of transiting planets per stellar system change
with metallicity?  (AKA: reproduce Weiss+18 CKS VI's Figure 1, the detected
system multiplicity function, for high and low metallicity CKS systems.)

--> Answer: hardly at all

4. We know that the period ratio P2/P1 tends to be larger than average when P1
is less than about 2-3 days. Perhaps this is due to tides. Scatter P2/P1 vs P1,
for different age quartiles. Do the same, for different metallicity quartiles.

--> Answer: there are some effects. (!)

USAGE
----------

$ python cks_multis_vs_age.py

'''
import matplotlib as mpl
#mpl.use('Agg')
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

from cks_age_exploration import _get_cks_data, arr


def _get_Weiss18_table1_stats(df):

    # number of transiting planets in sample
    N_tp = len(df)

    # number of systems (stars) in sample
    N_star = len(np.unique(df['id_starname']))

    # number of transiting planets in multiplanet systems in sample
    u, inv_inds, counts = np.unique(df['id_starname'],
                                    return_inverse=True,
                                    return_counts=True)
    multiplicity = counts[inv_inds]
    starnames = np.array(df['id_starname'])
    multis = starnames[multiplicity != 1]

    N_tpmulti = len(multis)

    return N_tp, N_star, N_tpmulti


def _apply_cks_VI_metallicity_study_filters(df):
    '''
    given df from _get_cks_data, return the boolean indices for the
    subset of interest.

    See Weiss+ 2018, CKS VI, table 1.

    Here we reproduce the "full CKS-Gaia multis sample".  We don't make the
    magnitude-limited cut, because we're gonna compare multis to multis.
    '''

    # row zero, `r0` tuple containing stats
    rows = []
    rows.append(_get_Weiss18_table1_stats(df))

    # not a false positive
    is_fp = arr(df['cks_fp'])
    sel = ~is_fp
    rows.append(_get_Weiss18_table1_stats(df[sel]))

    # radius correction factor <5%, logical or not given in Furlan+2017 table.
    sel &= ( (arr(df['fur17_rcorr_avg']) < 1.05) | (~np.isfinite(df['fur17_rcorr_avg']) ) )
    rows.append(_get_Weiss18_table1_stats(df[sel]))

    # not grazing (b<0.9)
    sel &= np.isfinite(arr(df['koi_impact']))
    sel &= arr(df['koi_impact']) < 0.9
    rows.append(_get_Weiss18_table1_stats(df[sel]))

    # SNR>10
    sel &= np.isfinite(arr(df['koi_model_snr']))
    sel &= arr(df['koi_model_snr']) > 10.
    rows.append(_get_Weiss18_table1_stats(df[sel]))

    # Rp < 22.4 Re
    sel &= np.isfinite(arr(df['giso_prad']))
    sel &= arr(df['giso_prad']) < 22.4
    rows.append(_get_Weiss18_table1_stats(df[sel]))

    # You need finite ages, too
    sel &= np.isfinite(arr(df['giso_slogage']))
    rows.append(_get_Weiss18_table1_stats(df[sel]))

    for row in rows:
        print(row)

    weiss_18_table_1 = [
        (1944,1222,1176),
        (1788,1118,1092),
        (1700,1060,1042),
        (1563,990,940),
        (1495,952,908),
        (1491,948,892),
    ]

    for rowind, wrow in enumerate(weiss_18_table_1):
        if wrow == rows[rowind]:
            print('row {:d} matches Weiss+18 Table 1'.format(rowind))
        elif np.abs(np.sum(wrow) - np.sum(rows[rowind])) < 10:
            print('row {:d} is close (w/in 10 planets) of Weiss+18 Table 1'.
                  format(rowind))
        else:
            print('row {:d} is quite different from Weiss+18 Table 1'.
                  format(rowind))

    return sel


def make_weiss18_VI_fig1(df, sel):
    # inds: indices that result in unique array of starnames
    # inv_inds: indices that go from unique array to reconstructed array
    # counts: number of times each unique item appears in u
    u, u_inds, inv_inds, counts = np.unique(df[sel]['id_starname'],
                                          return_index=True,
                                          return_inverse=True,
                                          return_counts=True)
    multiplicity_function = counts     # length: number of systems
    df_multiplicity = counts[inv_inds] # length: number of transiting planets
    multis = df[sel][df_multiplicity != 1] # length: number of planets in multi systems

    system_df = df[sel].iloc[u_inds] # length: number of systems. unique stellar properties here!

    # reproduce Weiss+18 CKS VI's Figure 1.
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,3))
    bins = np.arange(0.5,7.5,1)
    n, _, _ = ax.hist(counts[counts>=2], bins)
    for x,y,s in zip(bins[1:]-0.5, n+5, list(map(str,n.astype(int)))):
        ax.text(x,y,s, fontsize='xx-small', va='top')
    ax.set_xlabel('Number of Transiting Planets', fontsize='large')
    ax.set_xlim([0.5,7.5])
    ax.set_ylabel('Number of Stars', fontsize='large')
    ax.set_title('reproduce Weiss+18 CKS VI fig 1',
                 fontsize='xx-small')
    f.tight_layout()
    f.savefig('../results/cks_multis_vs_age/cks_gaia_multis_multiplicity_fn.pdf')


def split_weiss18_fig1_high_low_met(df, sel):
    # reproduce Weiss+18 CKS VI's Figure 1, the detected system multiplicity
    # function, for high and low metallicity CKS systems.

    u, u_inds, inv_inds, counts = np.unique(df[sel]['id_starname'],
                                          return_index=True,
                                          return_inverse=True,
                                          return_counts=True)
    multiplicity_function = counts     # length: number of systems
    df_multiplicity = counts[inv_inds] # length: number of transiting planets
    multis = df[sel][df_multiplicity != 1] # length: number of planets in multi systems

    system_df = df[sel].iloc[u_inds] # length: number of systems. unique stellar properties here!

    # get metallicities for systems with at least two planets
    smet_VII = arr(system_df['cks_smet'])[(counts>=2)]
    med_smet = np.median(smet_VII)

    sel_highmet = (smet_VII > med_smet)
    bins = np.arange(0.5,7.5,1)
    n_highmet, _ = np.histogram(counts[(counts>=2)][sel_highmet], bins)

    sel_lowmet = (smet_VII <= med_smet)
    n_lowmet, _ = np.histogram(counts[(counts>=2)][sel_lowmet], bins)

    N = int(np.max(counts))

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,3))

    ind = np.arange(1,N,1)
    width = 0.35

    rects1 = ax.bar(ind, n_highmet, width)
    rects2 = ax.bar(ind+width, n_lowmet, width)

    ax.set_xlabel('Number of Transiting Planets', fontsize='large')
    ax.set_xlim([0.5,7.5])
    ax.set_ylabel('Number of Stars', fontsize='large')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6'))

    ax.legend((rects1[0], rects2[0]),
              ('[Fe/H]$>${:.3f}'.format(med_smet),
               '[Fe/H]$\le${:.3f}'.format(med_smet)),
              fontsize='xx-small')

    ax.set_title('split CKS+Gaia multis by host star metallicity',
                 fontsize='xx-small')

    def autolabel(rects):
        #Attach a text label above each bar displaying its height
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom', fontsize='xx-small')

    autolabel(rects1)
    autolabel(rects2)

    f.tight_layout()
    f.savefig('../results/cks_multis_vs_age/cks_gaia_multis_multiplicity_highvlowmet.pdf')

def get_all_planet_pairs(df, sel):
    # we want all adjacent planet pairs
    # their radii, host star mass, semimajor axes, and the total number of
    # planets in the system.

    d = {'pair_inner_radius':[],
          'pair_outer_radius':[],
          's_mass':[],
          's_met':[],
          's_logage':[],
          'pair_inner_sma': [],
          'pair_outer_sma': [],
          'n_planet_in_sys':[],
          'pair_ind':[],
          'pair_inner_period':[],
          'pair_outer_period':[]
        }

    pair_ind = 0
    for sname in np.unique(df[sel]['id_starname']):

        this_sys = df[sel][ df[sel]['id_starname'] == sname ]

        if len(this_sys) == 1:
            continue

        # smallest semimaj axis at top. use gaia+cks+isochrone constrained vals.
        this_sys = this_sys.sort_values('giso_sma').reset_index()

        N_adj_pairs = len(this_sys) - 1

        for ix, adj_pair in enumerate(range(N_adj_pairs)):

            d['s_mass'].append(
                float(np.unique(this_sys['giso_smass']))
            )
            d['s_met'].append(
                float(np.unique(this_sys['cks_smet']))
            )
            d['s_logage'].append(
                float(np.unique(this_sys['giso_slogage']))
            )
            d['n_planet_in_sys'].append(
                int(len(this_sys))
            )
            d['pair_inner_radius'].append(
                float(this_sys.ix[ix]['giso_prad'])
            )
            d['pair_outer_radius'].append(
                float(this_sys.ix[ix+1]['giso_prad'])
            )
            d['pair_inner_sma'].append(
                float(this_sys.ix[ix]['giso_sma'])
            )
            d['pair_outer_sma'].append(
                float(this_sys.ix[ix+1]['giso_sma'])
            )
            d['pair_ind'].append(
                pair_ind
            )
            d['pair_inner_period'].append(
                float(this_sys.ix[ix]['koi_period'])
            )
            d['pair_outer_period'].append(
                float(this_sys.ix[ix+1]['koi_period'])
            )

            pair_ind += 1

    print('got planet pairs')
    pairs = pd.DataFrame(d)

    return pairs


def get_system_innermost_sma_by_Rstar(df, sel):

    d = {'sys_innermost_sma_by_rstar':[],
          's_mass':[],
          's_met':[],
          's_logage':[],
          'n_planet_in_sys':[]}

    for sname in np.unique(df[sel]['id_starname']):

        this_sys = df[sel][ df[sel]['id_starname'] == sname ]

        if len(this_sys) == 1:
            continue

        # smallest semimaj axis at top. use gaia+cks+isochrone constrained vals.
        this_sys = this_sys.sort_values('giso_sma').reset_index()

        N_adj_pairs = len(this_sys) - 1

        d['s_mass'].append(
            float(np.unique(this_sys['giso_smass']))
        )
        d['s_met'].append(
            float(np.unique(this_sys['cks_smet']))
        )
        d['s_logage'].append(
            float(np.unique(this_sys['giso_slogage']))
        )
        d['n_planet_in_sys'].append(
            int(len(this_sys))
        )
        d['sys_innermost_sma_by_rstar'].append(
            ((float(np.min(this_sys['giso_sma']))*u.AU) /
            (float(np.unique(this_sys['giso_srad']))*u.Rsun)).cgs.value
        )

    print('got system innermost sma/Rstar')
    systems = pd.DataFrame(d)

    return systems


def _get_WM14_mass(Rp):
    # Rp given in earth radii. (float)
    # Mp returned in earth masses (float)
    # Weiss & Marcy 2014, Weiss+ 2013.
    # see Eqs 5,6,7,8 from Weiss+ 2018, CKS VI

    R_p = Rp*u.Rearth

    if R_p < 1.5*u.Rearth:
        ρ_p = (2.43 + 3.39*(R_p.to(u.Rearth).value))*(u.g/(u.cm**3))
        M_p = (ρ_p/(5.51*(u.g/(u.cm**3)))) * \
                (R_p.to(u.Rearth).value)**3 * u.Mearth

    elif R_p >= 1.5*u.Rearth and R_p <= 4.*u.Rearth:
        M_p = 2.69*(R_p.to(u.Rearth).value)**(0.93) * u.Mearth

    elif R_p > 4.*u.Rearth and R_p < 9.*u.Rearth:
        M_p = 0.86*(R_p.to(u.Rearth).value)**(1.89) * u.Mearth

    elif R_p > 9.*u.Rearth:
        M_p = 318*u.Mearth

    return float(M_p.to(u.Mearth).value)


def compute_pair_separations_hill_radii(pairs):
    '''
    d = {'pair_inner_radius':[],
          'pair_outer_radius':[],
          's_mass':[],
          's_met':[],
          's_logage':[],
          'pair_inner_sma': [],
          'pair_outer_sma': [],
          'n_planet_in_sys':[],
          'pair_ind':[]}
    '''
    inner_masses, outer_masses = [], []
    for inner_rad, outer_rad in zip(
        arr(pairs['pair_inner_radius']), arr(pairs['pair_outer_radius'])):

        inner_masses.append(_get_WM14_mass(inner_rad))
        outer_masses.append(_get_WM14_mass(outer_rad))

    pairs['pair_inner_mass'] = arr(inner_masses)
    pairs['pair_outer_mass'] = arr(outer_masses)

    m_j = arr(pairs['pair_inner_mass'])*u.Mearth
    m_jp1 = arr(pairs['pair_outer_mass'])*u.Mearth
    M_star = arr(pairs['s_mass'])*u.Msun
    a_j = arr(pairs['pair_inner_sma'])*u.au
    a_jp1 = arr(pairs['pair_outer_sma'])*u.au

    R_H = 0.5 * ((m_j + m_jp1)/(3*M_star))**(1/3) * (a_j + a_jp1)
    sep = (a_jp1 - a_j)/R_H

    pairs['R_H'] = R_H.to(u.au).value
    pairs['sep_by_RH'] = sep.cgs.value

    return pairs


def make_weiss17_V_fig6(pairs):
    # reproduce Weiss+18 CKS VI's Figure 1.
    plt.close('all')
    f,axs = plt.subplots(nrows=2, ncols=1, figsize=(4,3), sharex=True)
    bins = np.arange(0,80+2.5,2.5)

    ax = axs.flatten()[0]

    n, _, _ = ax.hist(pairs['sep_by_RH'], bins, histtype='step')

    ax.set_xlim([0,80])
    ax.set_ylim([0,70])
    ax.set_ylabel('Number of planet pairs', fontsize='xx-small')
    ax.set_title('reproduce Weiss+18 CKS V fig 6',
                 fontsize='xx-small')
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.grid(True, zorder=-2, which='major', alpha=0.5)

    ax = axs.flatten()[1]
    counts, bin_edges = np.histogram(pairs['sep_by_RH'], bins=len(bins)*3, normed=True)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf/cdf[-1])

    ax.set_ylabel('CDF', fontsize='xx-small')
    ax.set_ylim([0,1])
    ax.set_xlabel('Separation in mutual Hill radii', fontsize='xx-small')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(True, zorder=-2, which='major', alpha=0.5)

    f.tight_layout(h_pad=0, w_pad=0)
    f.savefig('../results/cks_multis_vs_age/cks_gaia_pairs_separation.pdf',
             bbox_inches='tight')


def split_weiss17_V_fig6_by_age(pairs):

    s_logage = pairs['s_logage']

    low_third = np.percentile(s_logage, 33.3333333)
    up_third = np.percentile(s_logage, 2*33.3333333)

    low_inds = (s_logage < low_third)
    mid_inds = (s_logage > low_third) & (s_logage < up_third)
    up_inds  = (s_logage > up_third)


    # weiss+17 fig 6, but for oldest third, middle third, bottom third.
    plt.close('all')
    f,axs = plt.subplots(nrows=2, ncols=1, figsize=(4,3), sharex=True)
    bins = np.arange(0,80+5,5)

    ax = axs.flatten()[0]

    n, _, _ = ax.hist(pairs['sep_by_RH'], bins, histtype='step', label='all CKS multis',
                      lw=0.5)

    ax.set_xlim([0,80])
    ax.set_ylim([0,45])
    ax.set_ylabel('Number of planet pairs', fontsize='xx-small')
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.grid(True, zorder=-2, which='major', alpha=0.2)

    ax = axs.flatten()[1]
    counts, bin_edges = np.histogram(pairs['sep_by_RH'], bins=len(bins)*10,
                                     normed=True)
    cdf = np.cumsum(counts)
    ax.plot(bin_edges[1:], cdf/cdf[-1], label='all CKS multis', lw=0.5)

    ax.set_ylabel('CDF', fontsize='xx-small')
    ax.set_ylim([0,1])
    ax.set_xlabel('Separation in mutual Hill radii', fontsize='xx-small')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(True, zorder=-2, which='major', alpha=0.2)

    for ix, inds in enumerate([low_inds, mid_inds, up_inds]):
        ids = ['youngest', 'middle', 'oldest']

        tpairs = pairs[inds]

        for ax_ix, ax in enumerate(axs):
            if ax_ix == 0:
                ax.hist(tpairs['sep_by_RH'], bins, histtype='step',
                        label=ids[ix], lw=0.5)
                ax.legend(loc='best', fontsize='xx-small')
                print(tpairs['sep_by_RH'].describe())
            else:
                counts, bin_edges = np.histogram(tpairs['sep_by_RH'],
                                                 bins=len(bins)*10, normed=True)
                cdf = np.cumsum(counts)
                ax.plot(bin_edges[1:], cdf/cdf[-1], label=ids[ix], lw=0.5)

                ax.legend(loc='upper left', fontsize=4)

    young_pairs = pairs[low_inds]
    old_pairs = pairs[up_inds]
    from scipy import stats
    D, p_value = stats.ks_2samp(
        young_pairs['sep_by_RH'],old_pairs['sep_by_RH'])
    txt = 'D={:.2f},p={:.4f}\nfor 2sample KS (young v old)'.format(D, p_value)
    ax.text(0.9, 0.1, txt,
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize='xx-small')

    f.tight_layout(h_pad=0, w_pad=0)
    f.savefig('../results/cks_multis_vs_age/cks_gaia_pairs_separation_byage.pdf',
             bbox_inches='tight')



def make_stackedhist_innermost_sma(systems):

    # log radius bins
    logbin_aoverRstar = np.logspace(0, 2.25, num=10)
    logbin_x = logbin_aoverRstar

    s_logage = systems['s_logage']
    ages = s_logage

    linbins = [np.percentile(s_logage, x) for x in range(0,100+25,25)]

    left_linbins, right_linbins = linbins[:-1], linbins[1:]
    linbin_intervals = list(zip(left_linbins, right_linbins))
    age_intervals = linbin_intervals

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,3))

    for ix, age_interval in enumerate(age_intervals):

        minage, maxage = min(age_interval), max(age_interval)

        print(ix, minage, maxage)

        sel = (ages > minage) & (ages < maxage)

        N_cks = len(systems['sys_innermost_sma_by_rstar'][sel])
        print(systems['sys_innermost_sma_by_rstar'][sel].describe())

        try:
            weights = np.ones_like(systems['sys_innermost_sma_by_rstar'][sel])/float(N_cks)

            # be sure to plot the fraction of planets in each bin. (not the
            # pdf, which has INTEGRAL normalized to 1).
            agestr = '{:.2f}<log(age)<{:.2f}, N={:d}'.format(
                     minage, maxage, int(N_cks))
            ax.hist(systems['sys_innermost_sma_by_rstar'][sel], bins=logbin_x,
                    histtype='step', weights=weights, label=agestr)
        except:
            print('caught err in ax.hist, getting out')

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    ax.legend(loc='best', fontsize=4)
    ax.set_xlabel('a/Rstar for innermost planet of CKS multi-planet systems',
                  fontsize='xx-small')
    ax.set_ylabel('fraction of systems in age bin', fontsize='xx-small')
    ax.set_xscale('log')

    savpath = '../results/cks_multis_vs_age/cks_gaia_multi_innermost_sma_by_rstar_age_quartiles.pdf'

    f.tight_layout(h_pad=0, w_pad=0)
    f.savefig(savpath, bbox_inches='tight')

    print('made {:s}'.format(savpath))


def scatter_p2byp1_vs_p1(pairs):
    # the period ratio P2/P1 tends to be larger than average when P1 is less
    # than about 2-3 days

    P1 = arr(pairs['pair_inner_period'])
    P2 = arr(pairs['pair_outer_period'])

    plt.close('all')
    f,ax = plt.subplots(figsize=(8,6))

    ax.scatter(P1, P2/P1, marker='o', s=5, c='#1f77b4', zorder=2)

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    ax.set_xlabel('$P_1$, inner period of planet pair [days]',
                  fontsize='medium')
    ax.set_ylabel('$P_2/P_1$, ratio of planet pair periods',
                  fontsize='medium')
    ax.set_xscale('log')
    ax.set_yscale('log')

    savpath = '../results/cks_multis_vs_age/scatter_p2byp1_vs_p1.pdf'

    f.tight_layout(h_pad=0, w_pad=0)
    f.savefig(savpath, bbox_inches='tight')

    print('made {:s}'.format(savpath))



def scatter_p2byp1_vs_p1_metallicity_percentiles(pairs):

    P1 = arr(pairs['pair_inner_period'])
    P2 = arr(pairs['pair_outer_period'])
    smet = arr(pairs['s_met'])
    slogage = arr(pairs['s_logage'])

    sep = 25
    smet_pctls = [np.percentile(smet, pct) for pct in np.arange(0,100+sep,sep)]

    slogage_pctls = [np.percentile(slogage, pct) for pct in
                     np.arange(0,100+25,25)]

    # plot pairs by stellar metallicity percentiles
    plt.close('all')
    f,axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True,
                         sharey=True)
    axs = axs.flatten()

    for ix, ax in enumerate(axs[::-1]):
        ax.scatter(P1, P2/P1, marker='o', s=5, c='lightgray', zorder=1)

        if ix == 0:
            sel = smet < smet_pctls[ix+1]
            textstr = 'Fe/H$<${:.2f}\ntot={:d},blue={:d}'.format(
                smet_pctls[ix+1],len(pairs),int(len(pairs)*sep/100))
        elif ix == len(axs)-1:
            sel = smet > smet_pctls[ix]
            textstr = 'Fe/H$>${:.2f}\ntot={:d},blue={:d}'.format(
                smet_pctls[ix],len(pairs),int(len(pairs)*sep/100))
        else:
            sel = smet >= smet_pctls[ix]
            sel &= smet < smet_pctls[ix+1]
            textstr = 'Fe/H=({:.2f},{:.2f})\ntot={:d},blue={:d}'.format(
                smet_pctls[ix],smet_pctls[ix+1],len(pairs),int(len(pairs)*sep/100))

        ax.scatter(P1[sel], P2[sel]/P1[sel],
                   marker='o', s=3, c='#1f77b4', zorder=2)

        ax.text(0.95, 0.95, textstr, horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                fontsize='xx-small')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.4,300])
        ax.set_xlim([0.9,102])
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    f.tight_layout(h_pad=0, w_pad=0)

    # set labels
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)
    plt.grid(False)
    plt.xlabel('$P_1$, inner period of planet pair [days]', fontsize='medium')
    plt.ylabel('$P_2/P_1$, ratio of planet pair periods', fontsize='medium')

    savpath = '../results/cks_multis_vs_age/scatter_p2byp1_vs_p1_metallicity_percentiles.pdf'

    f.tight_layout(h_pad=0, w_pad=0)
    f.savefig(savpath, bbox_inches='tight')

    print('made {:s}'.format(savpath))


def scatter_p2byp1_vs_p1_age_percentiles(pairs):

    P1 = arr(pairs['pair_inner_period'])
    P2 = arr(pairs['pair_outer_period'])
    smet = arr(pairs['s_met'])
    slogage = arr(pairs['s_logage'])

    sep = 25
    smet_pctls = [np.percentile(smet, pct) for pct in np.arange(0,100+sep,sep)]

    slogage_pctls = [np.percentile(slogage, pct) for pct in
                     np.arange(0,100+25,25)]

    # plot pairs by stellar age percentiles
    plt.close('all')
    f,axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharex=True,
                         sharey=True)
    axs = axs.flatten()

    for ix, ax in enumerate(axs):
        ax.scatter(P1, P2/P1, marker='o', s=5, c='lightgray', zorder=1)

        if ix == 0:
            sel = slogage < slogage_pctls[ix+1]
            textstr = 'logage$<${:.2f}\ntot={:d},blue={:d}'.format(
                slogage_pctls[ix+1],len(pairs),int(len(pairs)*sep/100))
        elif ix == len(axs)-1:
            sel = slogage > slogage_pctls[ix]
            textstr = 'logage$>${:.2f}\ntot={:d},blue={:d}'.format(
                slogage_pctls[ix],len(pairs),int(len(pairs)*sep/100))
        else:
            sel = slogage >= slogage_pctls[ix]
            sel &= slogage < slogage_pctls[ix+1]
            textstr = 'logage=({:.2f},{:.2f})\ntot={:d},blue={:d}'.format(
                slogage_pctls[ix],slogage_pctls[ix+1],len(pairs),int(len(pairs)*sep/100))

        ax.scatter(P1[sel], P2[sel]/P1[sel],
                   marker='o', s=3, c='#1f77b4', zorder=2)

        ax.text(0.95, 0.95, textstr, horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                fontsize='xx-small')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.4,300])
        ax.set_xlim([0.9,102])
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    f.tight_layout(h_pad=0, w_pad=0)

    # set labels
    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)
    plt.grid(False)
    plt.xlabel('$P_1$, inner period of planet pair [days]', fontsize='medium')
    plt.ylabel('$P_2/P_1$, ratio of planet pair periods', fontsize='medium')

    savpath = '../results/cks_multis_vs_age/scatter_p2byp1_vs_p1_age_percentiles.pdf'

    f.tight_layout(h_pad=0, w_pad=0)
    f.savefig(savpath, bbox_inches='tight')

    print('made {:s}'.format(savpath))



if __name__ == '__main__':

    make_fig1_and_split = False
    make_fig6_weiss17 = False
    make_stackedhist_innermost = False

    df = _get_cks_data()
    sel = _apply_cks_VI_metallicity_study_filters(df)
    pairs = get_all_planet_pairs(df, sel)
    pairs = compute_pair_separations_hill_radii(pairs)
    systems = get_system_innermost_sma_by_Rstar(df, sel)

    if make_fig1_and_split:
        # How does the average number of transiting planets per stellar system
        # change with metallicity? 
        make_weiss18_VI_fig1(df, sel)
        split_weiss18_fig1_high_low_met(df, sel)

    if make_fig6_weiss17:
        # Does average separation between planet pairs in multis (in units of
        # mutual hill radii increase for the oldest multis?
        make_weiss17_V_fig6(pairs)
        split_weiss17_V_fig6_by_age(pairs)

    if make_stackedhist_innermost:
        # Does the innermost planet of a multi tend to be detected further away
        # from the host star in the oldest systems?
        make_stackedhist_innermost_sma(systems)

    # Scatter P2/P1 vs P1, for different age quartiles.
    scatter_p2byp1_vs_p1(pairs)
    scatter_p2byp1_vs_p1_metallicity_percentiles(pairs)
    scatter_p2byp1_vs_p1_age_percentiles(pairs)
