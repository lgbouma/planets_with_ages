# -*- coding: utf-8 -*-
"""
first, reproduce Mazeh 2015's result

then, note if a/Rstar<5 KOIs (not in multiple planet systems) were more
misaligned, they would show less photometric variability than e.g., the
a/Rstar>5 KOIs.

this would be consistent with Dai+ 2018's result
"""
from __future__ import division, print_function

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, warnings, pickle

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from astropy.stats import median_absolute_deviation

from numpy import array as nparr

def get_Mazeh_2015_KOI_table():
    t = Table.read("../data/Mazeh_2015_KOI_table_1.vot", format="votable")
    print("got Mazeh15 table of length {:d}".format(len(t)))
    return t

def apply_Mazeh_2015_cuts(t):
    """
    C: "1" if Centroid motion shows transit and rotational
    modulation on different stars (meta.code)
    G: "1" if star is likely a giant
    T: "1" if Teff outside 3500-6500K
    F: "1" if false positive (EB) identified in this work
    R: "1" if rejected by visual examination stage
    M1: "1" if inconsistent detection in different quarters
    M2: "1" if inconsistent detection with not high enough peaks
    """
    df = t.to_pandas()

    sel = ~pd.isnull(df["Teff"])
    sel &= ~pd.isnull(df["Rvar"])
    sel &= ~df["C"].astype(int)
    sel &= ~df["G"].astype(int)
    sel &= ~df["T"].astype(int)
    sel &= ~df["F"].astype(int)
    sel &= ~df["R"].astype(int)
    sel &= ~df["M1"].astype(int)
    sel &= ~df["M2"].astype(int)

    print("after cuts, Mazeh15 table has length {:d}".format(len(df[sel])))
    if not len(df[sel])==993:
        raise AssertionError("should be same as Mazeh15 figure 1")
    return df[sel]

def get_McQuillan_2014_nonKOI_table():
    t = Table.read("../data/McQuillan_2014_ApJS_211_24_table1.vot",
                   format="votable")
    return t

def apply_McQuillan_2014_cuts(t):
    """
    From Mazeh15: The selection criteria for these (MMA14's) targets were
    described in detail in MMA14, and included the same Teff and log g cuts for
    main sequence as are used in the present paper, as well as removal of known
    KOIs and EBs. For the present work we constrain the Teff range to 3500–6500
    K, leaving a total sample of 33,614 single stars.

    MMA14 write:
    The initial list of all targets observed by Kepler contains ∼195,000 stars.
    To select only main-sequence stars, we used the Teff–log g and color–color
    cuts advocated by Ciardi et al. (2011), which remove ∼32,000 giants from
    the Kepler sample.  At this stage we also excluded ∼2000 targets without
    Teff or log g values. Where available for the low-mass stars, we use the
    improved values of Dressing & Charbonneau (2013) in place of the KIC
    parameters. A more recent version of the KIC is available (Huber et al.
    2014), however, we have opted to use the original since this provides the
    most homogeneous parameters.  We checked that using the updated KIC values
    does not alter the results of this paper.
    """
    df = t.to_pandas()

    sel = (3500 < t["Teff"])
    sel &= (t["Teff"] < 6500)

    print("after cuts, McQuillan14 table has length {:d}".format(len(df[sel])))
    if not len(df[sel])==33614:
        raise AssertionError("should be same as Mazeh15 figure 1")
    return df[sel]

def calculate_medians(df, Rvarkey="Rvar"):

    # calculation of median
    bin_edges = np.arange(3500, 6500+250, 250)
    teffbins = [(binmin, binmax) for binmin, binmax in
                zip(bin_edges[:-1], bin_edges[1:])]

    median_rvar, median_rvar_errs = [], []
    for teffbin in teffbins:
        sel = (df["Teff"] > min(teffbin))
        sel &= (df["Teff"] < max(teffbin))
        n_pts = len(df[sel])
        median_rvar.append(np.nanmedian(df[sel][Rvarkey]))
        median_rvar_errs.append(
            median_absolute_deviation(df[sel][Rvarkey])/np.sqrt(n_pts)
        )
    median_rvar = nparr(median_rvar)
    median_rvar_errs = nparr(median_rvar_errs)

    teff_bin_middles = np.arange(3625, 6500+125, 250)

    return (median_rvar, median_rvar_errs, teff_bin_middles)

def get_running_median(df, size, Rvarkey='Rvar'):

    df = df.sort_values('Teff')
    from scipy.ndimage.filters import median_filter
    mf_teff = median_filter(nparr(df[Rvarkey]), size=size, mode='reflect')

    return mf_teff, nparr(df['Teff'])

def make_Mazeh_2015_figure_1(df_koi, df_notkoi):
    """
    Divide each sample in 250K bins, calculate and plot median of each bin.
    Error on the median is MAD/sqrt(n) (Mazeh+ 2015).  (wikipedia) "Moreover,
    the MAD is a robust statistic, being more resilient to outliers in a data
    set than the standard deviation. In the standard deviation, the distances
    from the mean are squared, so large deviations are weighted more heavily,
    and thus outliers can heavily influence it. In the MAD, the deviations of a
    small number of outliers are irrelevant."
    """

    koi_median_rvar, koi_median_rvar_errs, teff_bin_middles = \
            calculate_medians(df_koi, Rvarkey="Rvar")
    notkoi_median_rvar, notkoi_median_rvar_errs, _ = \
            calculate_medians(df_notkoi, Rvarkey="Rper")

    # save the data ur about to plot
    df = pd.DataFrame({
        "koi_median_rvar":koi_median_rvar,
        "koi_median_rvar_errs":koi_median_rvar_errs,
        "notkoi_median_rvar":notkoi_median_rvar,
        "notkoi_median_rvar_errs":notkoi_median_rvar_errs,
        "teff_bin_middles":teff_bin_middles
    })
    savpath = "../data/Mazeh_2015_figure_1_median_values.csv"
    df.to_csv(savpath, index=False)
    print("saved {:s}".format(savpath))

    # "The figure also displays for each sample a running median, with 1000 and
    # 250 points width for the single and the KOIs amplitudes, which were then
    # smoothed with a width of 501 and 51 points, respectively."
    mf_rvar_koi, mf_teff_koi = get_running_median(df_koi, 250, Rvarkey='Rvar')
    mf_rvar_notkoi, mf_teff_notkoi = get_running_median(df_notkoi, 1000,
                                                        Rvarkey='Rper')

    mf_d = {
        'mf_rvar_koi':np.log10(mf_rvar_koi),
        'mf_teff_koi':mf_teff_koi,
        'mf_rvar_notkoi':np.log10(mf_rvar_notkoi),
        'mf_teff_notkoi':mf_teff_notkoi
    }
    mf_savpath = "../data/Mazeh_2015_figure_1_running_median_values.pickle"
    with open(mf_savpath, 'wb') as f:
        pickle.dump(mf_d, f, pickle.HIGHEST_PROTOCOL)
        print('saved {:s}'.format(mf_savpath))

    # make plot
    plt.close("all")
    f, ax = plt.subplots(figsize=(8,6))

    # background scatter points
    ax.scatter(df_koi["Teff"], df_koi["Rvar"], s=4, zorder=2, rasterized=True,
               label="{:d} KOIs".format(len(df_koi)), linewidths=0, c="red")
    ax.scatter(df_notkoi["Teff"], df_notkoi["Rper"], s=1, alpha=0.5, zorder=1,
               rasterized=True, label="{:d} nonKOIs".format(len(df_notkoi)),
               linewidths=0, c="gray")

    # 250 K bins medians and their errors
    ax.scatter(teff_bin_middles, koi_median_rvar, color="red", marker="o",
               linewidths=0, zorder=2, s=25, rasterized=True)
    ax.errorbar(teff_bin_middles, koi_median_rvar, yerr=koi_median_rvar_errs,
                elinewidth=0.3, ecolor="red", capsize=1, capthick=1,
                linewidth=1, fmt="s", ms=0, zorder=2, alpha=1)
    ax.scatter(teff_bin_middles, notkoi_median_rvar, color="gray", marker="o",
               linewidths=0, zorder=1, s=25, rasterized=True)
    ax.errorbar(teff_bin_middles, notkoi_median_rvar,
                yerr=notkoi_median_rvar_errs, elinewidth=0.3, ecolor="gray",
                capsize=1, capthick=1, linewidth=1, fmt="s", ms=0, zorder=1,
                alpha=1)

    # running median line
    minteff, maxteff = 3625, 6375
    sel_koi = (mf_teff_koi > minteff) & (mf_teff_koi < maxteff)
    sel_notkoi = (mf_teff_notkoi > minteff) & (mf_teff_notkoi < maxteff)
    ax.plot(mf_teff_koi[sel_koi], mf_rvar_koi[sel_koi], color='red',
            marker=None, linewidth=2, zorder=2, markersize=0)
    ax.plot(mf_teff_notkoi[sel_notkoi], mf_rvar_notkoi[sel_notkoi],
            color='gray', marker=None, linewidth=2, zorder=1, markersize=0)

    ax.set_yscale("log")
    ax.legend(loc="lower left")
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("rotation amplitude [ppm]")
    ax.set_ylim([10**(2.3), 10**(5.2)])

    f.tight_layout()
    savpath = ("../results/short_period_photometric_amplitude_test/"
               "Mazeh_2015_fig1_replication.png")
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))

def get_a_over_Rstar_and_singles_v_multis_columns(mazeh_df):

    # fix auto float typing
    mazeh_df['KIC'] = mazeh_df['KIC'].astype(int)

    exo_tab = Table.read(
        "../data/20180916_exoarchive_KOIs_for_Mazeh_xmatch.vot",
        format="votable")
    exo_df = exo_tab.to_pandas()

    sel_exo_df = exo_df[['koi_dor', 'kepid', 'koi_count']]
    sel_exo_df['kepid'] = sel_exo_df['kepid'].astype(int)

    # Note that Mazeh counted each KOI, even if it was a multiple, only ONCE. 
    # This makes sense in a plot of variability amplitue vs Teff, because these
    # are STELLAR properties.

    # So we want to do two different merges. One to get Mazeh's exact same
    # star table, but with the koi_count column. The other to get a planet
    # table, with a koi_dor column as well.

    planet_df = pd.merge(mazeh_df, sel_exo_df, how='left', left_on='KIC',
                         right_on='kepid')
    print('{:d} planets total'.format(len(planet_df)))
    print('{:d} planets with a/Rstar<5'.format(
        len(planet_df[planet_df['koi_dor']<5])))

    # don't include koi_dor, because this is a DataFrame for stars
    sel_star_df = sel_exo_df.drop_duplicates('kepid')
    sel_star_df = sel_star_df[['kepid','koi_count']]
    star_df = pd.merge(mazeh_df, sel_star_df,
                       how='left', left_on='KIC', right_on='kepid')
    if not len(star_df)==993:
        raise AssertionError("should be same as Mazeh15 figure 1")

    return planet_df, star_df

def plot_varamplitude_vs_teff_koi_v_notkoi(df_koi, df_notkoi):

    sel_single = (df_koi['koi_count']==1)
    sel_multi = (df_koi['koi_count']>1)
    singlekoi_median_rvar, singlekoi_median_rvar_errs, teff_bin_middles = \
            calculate_medians(df_koi[sel_single], Rvarkey="Rvar")
    multikoi_median_rvar, multikoi_median_rvar_errs, _ = \
            calculate_medians(df_koi[sel_multi], Rvarkey="Rvar")
    notkoi_median_rvar, notkoi_median_rvar_errs, _ = \
            calculate_medians(df_notkoi, Rvarkey="Rper")

    # save the data to what you're about to plot
    df = pd.DataFrame({
        "singlekoi_median_rvar":singlekoi_median_rvar,
        "singlekoi_median_rvar_errs":singlekoi_median_rvar_errs,
        "multikoi_median_rvar":multikoi_median_rvar,
        "multikoi_median_rvar_errs":multikoi_median_rvar_errs,
        "notkoi_median_rvar":notkoi_median_rvar,
        "notkoi_median_rvar_errs":notkoi_median_rvar_errs,
        "teff_bin_middles":teff_bin_middles
    })
    savpath = "../data/varamplitude_vs_teff_koi_v_notkoi.csv"
    df.to_csv(savpath, index=False)
    print("saved {:s}".format(savpath))

    # make plot
    plt.close("all")
    f, ax = plt.subplots(figsize=(8,6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # first do the scatter points
    ax.scatter(df_koi[sel_single]["Teff"], df_koi[sel_single]["Rvar"], s=5,
               zorder=2, rasterized=True, linewidths=0, c=colors[0],
               label="{:d} single KOIs".format(len(df_koi[sel_single])))
    ax.scatter(df_koi[sel_multi]["Teff"], df_koi[sel_multi]["Rvar"], s=5,
               zorder=3, rasterized=True, linewidths=0, c=colors[1],
               label="{:d} multi KOIs".format(len(df_koi[sel_multi])))
    ax.scatter(df_notkoi["Teff"], df_notkoi["Rper"], s=1, alpha=0.5, zorder=1,
               rasterized=True, label="{:d} nonKOIs".format(len(df_notkoi)),
               linewidths=0, c="gray")

    # then plot the medians
    ax.plot(teff_bin_middles, singlekoi_median_rvar, color=colors[0], marker="o",
            linewidth=2, zorder=5, markersize=5)
    ax.errorbar(teff_bin_middles, singlekoi_median_rvar,
                yerr=singlekoi_median_rvar_errs, elinewidth=0.3,
                ecolor=colors[0], capsize=1, capthick=1, linewidth=1, fmt="s",
                ms=0, zorder=5)
    ax.plot(teff_bin_middles, multikoi_median_rvar, color=colors[1],
            marker="o", linewidth=2, zorder=6, markersize=5)
    ax.errorbar(teff_bin_middles, multikoi_median_rvar,
                yerr=multikoi_median_rvar_errs, elinewidth=0.3,
                ecolor=colors[1], capsize=1, capthick=1, linewidth=1, fmt="s",
                ms=0, zorder=6)
    ax.plot(teff_bin_middles, notkoi_median_rvar, color="gray", marker="o",
            linewidth=2, zorder=4, markersize=5)
    ax.errorbar(teff_bin_middles, notkoi_median_rvar,
                yerr=notkoi_median_rvar_errs, elinewidth=0.3, ecolor="gray",
                capsize=1, capthick=1, linewidth=1, fmt="s", ms=0, zorder=4)

    ax.set_yscale("log")

    ax.legend(loc="lower left")

    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("rotation amplitude [ppm]")
    ax.set_ylim([10**(2.3), 10**(5.2)])

    f.tight_layout()
    savpath = ("../results/short_period_photometric_amplitude_test/"
               "varamplitude_vs_teff_koi_v_notkoi.png")
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))


def plot_varamplitude_vs_teff_singles_close_far(planet_df, df_notkoi, dor_cut):

    sel_single = (planet_df['koi_count']==1)
    sel_close_single = (planet_df['koi_dor'] <= dor_cut) & sel_single
    sel_far_single = (planet_df['koi_dor'] > dor_cut) & sel_single

    closekoi_median_rvar, closekoi_median_rvar_errs, teff_bin_middles = \
            calculate_medians(planet_df[sel_close_single], Rvarkey="Rvar")
    farkoi_median_rvar, farkoi_median_rvar_errs, _ = \
            calculate_medians(planet_df[sel_far_single], Rvarkey="Rvar")
    notkoi_median_rvar, notkoi_median_rvar_errs, _ = \
            calculate_medians(df_notkoi, Rvarkey="Rper")

    # save the data to what you're about to plot
    df = pd.DataFrame({
        "closekoi_median_rvar":closekoi_median_rvar,
        "closekoi_median_rvar_errs":closekoi_median_rvar_errs,
        "farkoi_median_rvar":farkoi_median_rvar,
        "farkoi_median_rvar_errs":farkoi_median_rvar_errs,
        "notkoi_median_rvar":notkoi_median_rvar,
        "notkoi_median_rvar_errs":notkoi_median_rvar_errs,
        "teff_bin_middles":teff_bin_middles
    })
    savpath = ("../data/varamplitude_vs_teff_singles_close_far"
               "_dorcut{:.1f}.csv".format(dor_cut))
    df.to_csv(savpath, index=False)
    print("saved {:s}".format(savpath))

    # make plot
    plt.close("all")
    f, ax = plt.subplots(figsize=(8,6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # first do the scatter points
    ax.scatter(planet_df[sel_close_single]["Teff"],
               planet_df[sel_close_single]["Rvar"], s=5, zorder=2,
               rasterized=True, linewidths=0, c=colors[0],
               label="{:d} close single KOIs".format(
               len(planet_df[sel_close_single])))

    ax.scatter(planet_df[sel_far_single]["Teff"],
               planet_df[sel_far_single]["Rvar"], s=5, zorder=3,
               rasterized=True, linewidths=0, c=colors[1],
               label="{:d} far single KOIs".format(
               len(planet_df[sel_far_single])))

    ax.scatter(df_notkoi["Teff"], df_notkoi["Rper"], s=1, alpha=0.5, zorder=1,
               rasterized=True, label="{:d} nonKOIs".format(len(df_notkoi)),
               linewidths=0, c="gray")

    # then plot the medians
    ax.plot(teff_bin_middles, closekoi_median_rvar, color=colors[0],
            marker="o", linewidth=2, zorder=5, markersize=5)
    ax.errorbar(teff_bin_middles, closekoi_median_rvar,
                yerr=closekoi_median_rvar_errs, elinewidth=0.3,
                ecolor=colors[0], capsize=1, capthick=1, linewidth=1, fmt="s",
                ms=0, zorder=5)

    ax.plot(teff_bin_middles, farkoi_median_rvar, color=colors[1], marker="o",
            linewidth=2, zorder=6, markersize=5)
    ax.errorbar(teff_bin_middles, farkoi_median_rvar,
                yerr=farkoi_median_rvar_errs, elinewidth=0.3, ecolor=colors[1],
                capsize=1, capthick=1, linewidth=1, fmt="s", ms=0, zorder=6)

    ax.plot(teff_bin_middles, notkoi_median_rvar, color="gray", marker="o",
            linewidth=2, zorder=4, markersize=5)
    ax.errorbar(teff_bin_middles, notkoi_median_rvar,
                yerr=notkoi_median_rvar_errs, elinewidth=0.3, ecolor="gray",
                capsize=1, capthick=1, linewidth=1, fmt="s", ms=0, zorder=4)

    ax.set_yscale("log")

    ax.legend(loc="lower left")

    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("rotation amplitude [ppm]")
    ax.set_ylim([10**(2.3), 10**(5.2)])

    f.tight_layout()
    savpath = ("../results/short_period_photometric_amplitude_test/"
               "varamplitude_vs_teff_singles_close_far"
               "_dorcut{:.1f}.png".format(dor_cut))
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))


def make_Mazeh_2015_figure_2(df_koi, df_notkoi):

    f, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,6))

    sel_df_koi_cold = (df_koi["Teff"]>3500) & (df_koi["Teff"]<6000)
    sel_df_koi_hot = (df_koi["Teff"]>6000) & (df_koi["Teff"]<6500)
    sel_df_notkoi_cold = (df_notkoi["Teff"]>3500) & (df_notkoi["Teff"]<6000)
    sel_df_notkoi_hot = (df_notkoi["Teff"]>6000) & (df_notkoi["Teff"]<6500)

    teffstrs = ['teff=3500-6000K','teff=6000-6500K']

    # cold, then hot.
    for ax, sel_koi, sel_notkoi, teffstr in zip(
    axs, [sel_df_koi_cold, sel_df_koi_hot],
    [sel_df_notkoi_cold, sel_df_notkoi_hot], teffstrs):

        counts, bin_edges = np.histogram(np.log10(df_koi[sel_koi]['Rvar']),
                                         bins=len(df_koi[sel_koi]),
                                         normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1],
                label='{:d} KOIs'.format(len(df_koi[sel_koi])), lw=0.5)

        counts, bin_edges = np.histogram(np.log10(df_notkoi[sel_notkoi]['Rper']),
                                         bins=len(df_notkoi[sel_notkoi]),
                                         normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1],
                label='{:d} nonKOIs'.format(len(df_notkoi[sel_notkoi])),
                lw=0.5)

        ax.legend(loc='upper left', fontsize='xx-small')

        from scipy import stats
        D, p_value = stats.ks_2samp(np.log10(df_koi[sel_koi]['Rvar']),
                                    np.log10(df_notkoi[sel_notkoi]['Rper']))
        txt = '{:s}\nD={:.2e},p={:.2e} for 2sample KS'.format(teffstr, D, p_value)
        ax.text(0.95, 0.05, txt,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize='xx-small')

        ax.set_xlabel('log(amplitude [ppm])')

    # save it
    f.tight_layout()
    savpath = ("../results/short_period_photometric_amplitude_test/"
               "Mazeh_2015_fig2_replication.png")
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))


def plot_ks2sample_varamplitude_koi_v_notkoi(df_koi, df_notkoi):

    f, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,6))

    sel_df_koi_cold = (df_koi["Teff"]>3500) & (df_koi["Teff"]<5800)
    sel_df_koi_hot = (df_koi["Teff"]>5800) & (df_koi["Teff"]<6500)
    sel_df_notkoi_cold = (df_notkoi["Teff"]>3500) & (df_notkoi["Teff"]<5800)
    sel_df_notkoi_hot = (df_notkoi["Teff"]>5800) & (df_notkoi["Teff"]<6500)

    sel_single = (df_koi['koi_count']==1)
    sel_multi = (df_koi['koi_count']>1)

    teffstrs = ['teff=3500-5800K','teff=5800-6500K']

    # cold, then hot.
    for ax, sel_koi, sel_notkoi, teffstr in zip(
    axs, [sel_df_koi_cold, sel_df_koi_hot],
    [sel_df_notkoi_cold, sel_df_notkoi_hot], teffstrs):

        counts, bin_edges = np.histogram(np.log10(df_koi[sel_koi&sel_single]['Rvar']),
                                         bins=len(df_koi[sel_koi&sel_single]),
                                         normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1],
                label='{:d} single KOIs'.format(len(df_koi[sel_koi&sel_single])), lw=0.5)

        counts, bin_edges = np.histogram(np.log10(df_koi[sel_koi&sel_multi]['Rvar']),
                                         bins=len(df_koi[sel_koi&sel_multi]),
                                         normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1],
                label='{:d} multi KOIs'.format(len(df_koi[sel_koi&sel_multi])), lw=0.5)

        counts, bin_edges = np.histogram(np.log10(df_notkoi[sel_notkoi]['Rper']),
                                         bins=len(df_notkoi[sel_notkoi]),
                                         normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1],
                label='{:d} nonKOIs'.format(len(df_notkoi[sel_notkoi])),
                lw=0.5)

        ax.legend(loc='upper left', fontsize='xx-small')

        from scipy import stats
        D, p_value = stats.ks_2samp(np.log10(df_koi[sel_koi&sel_single]['Rvar']),
                                    np.log10(df_notkoi[sel_notkoi]['Rper']))
        D_mn, p_value_mn = stats.ks_2samp(np.log10(df_koi[sel_koi&sel_multi]['Rvar']),
                                    np.log10(df_notkoi[sel_notkoi]['Rper']))
        D_sm, p_value_sm = stats.ks_2samp(np.log10(df_koi[sel_koi&sel_single]['Rvar']),
                                          np.log10(df_koi[sel_koi&sel_multi]['Rvar']))
        txt = (
            '{:s}\nD={:.1e},p={:.1e} for KS single vs notKOI'.format(
                teffstr, D, p_value)+
            '\nD={:.1e},p={:.1e} for KS multi vs notKOI'.format(
                D_mn, p_value_mn)+
            '\nD={:.1e},p={:.1e} for KS single vs multi'.format(
                D_sm, p_value_sm)
        )
        ax.text(0.95, 0.05, txt,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize='xx-small')

        ax.set_xlabel('log(amplitude [ppm])')

    # save it
    f.tight_layout()
    savpath = ("../results/short_period_photometric_amplitude_test/"
               "ks2sample_varamplitude_koi_v_notkoi.png")
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))


def plot_ks2sample_varamplitude_close_far(df_koi, df_notkoi, dor_cut):

    f, axs = plt.subplots(nrows=1,ncols=2,figsize=(8,6))

    # everything is singles!!
    sel_single = (planet_df['koi_count']==1)
    sel_close = (planet_df['koi_dor'] <= dor_cut) & sel_single
    sel_far = (planet_df['koi_dor'] > dor_cut) & sel_single

    sel_df_koi_cold = (df_koi["Teff"]>3500) & (df_koi["Teff"]<5800) & sel_single
    sel_df_koi_hot = (df_koi["Teff"]>5800) & (df_koi["Teff"]<6500) & sel_single
    sel_df_notkoi_cold = (df_notkoi["Teff"]>3500) & (df_notkoi["Teff"]<5800)
    sel_df_notkoi_hot = (df_notkoi["Teff"]>5800) & (df_notkoi["Teff"]<6500)

    teffstrs = ['teff=3500-5800K','teff=5800-6500K']

    # cold, then hot.
    for ax, sel_koi, sel_notkoi, teffstr in zip(
    axs, [sel_df_koi_cold, sel_df_koi_hot],
    [sel_df_notkoi_cold, sel_df_notkoi_hot], teffstrs):

        # single KOI and close
        counts, bin_edges = np.histogram(np.log10(df_koi[sel_koi&sel_close]['Rvar']),
                                         bins=len(df_koi[sel_koi&sel_close]),
                                         normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1],
                label='{:d} close single KOIs'.format(len(df_koi[sel_koi&sel_close])), lw=0.5)

        # single KOI and far
        counts, bin_edges = np.histogram(np.log10(df_koi[sel_koi&sel_far]['Rvar']),
                                         bins=len(df_koi[sel_koi&sel_far]),
                                         normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1],
                label='{:d} far single KOIs'.format(len(df_koi[sel_koi&sel_far])), lw=0.5)

        # not KOI
        counts, bin_edges = np.histogram(np.log10(df_notkoi[sel_notkoi]['Rper']),
                                         bins=len(df_notkoi[sel_notkoi]),
                                         normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf/cdf[-1],
                label='{:d} nonKOIs'.format(len(df_notkoi[sel_notkoi])),
                lw=0.5)

        ax.legend(loc='upper left', fontsize='xx-small')

        from scipy import stats
        D_cn, p_value_cn = stats.ks_2samp(np.log10(df_koi[sel_koi&sel_close]['Rvar']),
                                    np.log10(df_notkoi[sel_notkoi]['Rper']))
        D_fn, p_value_fn = stats.ks_2samp(np.log10(df_koi[sel_koi&sel_far]['Rvar']),
                                    np.log10(df_notkoi[sel_notkoi]['Rper']))
        D_cf, p_value_cf = stats.ks_2samp(np.log10(df_koi[sel_koi&sel_close]['Rvar']),
                                          np.log10(df_koi[sel_koi&sel_far]['Rvar']))
        txt = (
            '{:s}\nD={:.1e},p={:.1e} for KS close vs notKOI'.format(
                teffstr, D_cn, p_value_cn)+
            '\nD={:.1e},p={:.1e} for KS far vs notKOI'.format(
                D_fn, p_value_fn)+
            '\nD={:.1e},p={:.1e} for KS close vs far'.format(
                D_cf, p_value_cf)
        )
        ax.text(0.95, 0.05, txt,
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize='xx-small')

        ax.set_xlabel('log(amplitude [ppm])')

    # save it
    f.tight_layout()
    savpath = ("../results/short_period_photometric_amplitude_test/"
               "ks2sample_varamplitude_close_far_dorcut{:.1f}.png".format(dor_cut))
    f.savefig(savpath, bbox_inches="tight", dpi=350)
    print("made {:s}".format(savpath))



if __name__ == "__main__":
    warnings.simplefilter("ignore", category=AstropyWarning)
    t = get_Mazeh_2015_KOI_table()
    df_koi = apply_Mazeh_2015_cuts(t)
    n = get_McQuillan_2014_nonKOI_table()
    df_notkoi = apply_McQuillan_2014_cuts(n)

    make_Mazeh_2015_figure_1(df_koi, df_notkoi)
    make_Mazeh_2015_figure_2(df_koi, df_notkoi)

    # 1) Recall Morton & Winn 2014, who used to vsini method and a hierarchical
    # bayesian model to infer that singles have higher obliquities than multis.
    # If this applies for Mazeh's sample, singles should have photometric
    # rotation amplitudes closer to isotropic ("nonKOIs") than multis.
    planet_df, star_df = get_a_over_Rstar_and_singles_v_multis_columns(df_koi)
    plot_varamplitude_vs_teff_koi_v_notkoi(star_df, df_notkoi)
    plot_ks2sample_varamplitude_koi_v_notkoi(star_df, df_notkoi)

    # 2) Recall Dai et al 2018, who showed a/Rstar<5 planets in multis have
    # higher-than-normal mutual inclinations. If the generalizaiton of this
    # applies to Mazeh's sample, a/Rstar<5 SINGLES should have photometric
    # rotation amplitudes closer to isotropic ("nonKOIs") than a/Rstar>5
    # singles.
    plot_varamplitude_vs_teff_singles_close_far(planet_df, df_notkoi, 5)
    plot_ks2sample_varamplitude_close_far(planet_df, df_notkoi, 5)
    plot_varamplitude_vs_teff_singles_close_far(planet_df, df_notkoi, 7.5)
    plot_ks2sample_varamplitude_close_far(planet_df, df_notkoi, 7.5)
    plot_varamplitude_vs_teff_singles_close_far(planet_df, df_notkoi, 10)
    plot_ks2sample_varamplitude_close_far(planet_df, df_notkoi, 10)
