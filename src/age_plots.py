# -*- coding: utf-8 -*-
'''
scatter,histogram,quartiles for SD18, exoarchive, & CKS

----------

plot_wellmeasuredparam:
    scatter x vs age with errors on age (y)

plot_jankyparam
    scatter x vs age with errors on both x and age

make_age_histograms_get_f_inds
    histograms of ages, and age/error_age

make_stacked_histograms
    histograms of radius, and period -- but at various time bins

make_quartile_scatter
    reproduce Fig 4 of Petigura+ 2018, but in age quartiles

make_boxplot_koi_count
    a seaborn "whisker" plot of koi number vs age
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

import os

def arr(x):
    return np.array(x)

def plot_wellmeasuredparam(tab, finite_age_inds, xparam, logx, logy,
                           is_exoarchive=False, is_cks=False,
                           is_sandersdas=False, savdir_append=''):
    '''
    args:
        tab (DataFrame or astropy table)
        finite_age_inds
        xparam (str): thing you want to scatter plot again age

    kwargs:
        only one of is_exoarchive, is_cks, is_sandersdas should be true.
    '''

    goodvals = tab[xparam][finite_age_inds]

    # only one of is_exoarchive, is_cks, or is_sandersdas should be True.
    assert np.sum(np.array([(is_exoarchive != is_cks),
                  (is_cks != is_sandersdas),
                  (is_exoarchive != is_sandersdas)]).astype(int)) == 2

    if is_exoarchive:
        ages = tab['st_age'][finite_age_inds]
        ages_perr = tab['st_ageerr1'][finite_age_inds]
        ages_merr = np.abs(tab['st_ageerr2'][finite_age_inds])
    elif is_cks:
        ages = 10**(tab['giso_slogage'][finite_age_inds])
        ages_perr = 10**(tab['giso_slogage_err1'][finite_age_inds])
        ages_merr = 10**(np.abs(tab['giso_slogage_err2'][finite_age_inds]))
    elif is_sandersdas:
        ages = 10**(tab['log10_age'][finite_age_inds])
        ages_perr = 10**(tab['log10_age_err'][finite_age_inds])
        ages_merr = 10**(np.abs(tab['log10_age_err'][finite_age_inds]))
    else:
        raise NotImplementedError

    # only plot planets with P < 1e4, or Rp<2.5Rj. this requires some
    # sick nparray formation tricks.
    xsel = -1
    if 'pl_orbper' in xparam:
        xsel = arr(goodvals) < 1e4
    if 'pl_radj' in xparam:
        xsel = arr(goodvals) < 2.5
    if 'pl_rade' in xparam:
        xsel = arr(goodvals) < 28
    if 'giso_prad' in xparam and is_cks:
        xsel = arr(goodvals) < 3e1
    if 'koi_period' in xparam:
        xsel = arr(goodvals) < 2e2
    if 'koi_dor' in xparam:
        xsel = arr(goodvals) < 178


    if type(xsel) != int:
        # you selected by some criterion in xparam
        xvals, yvals = arr(goodvals[xsel]), arr(ages[xsel])
        ages_perr = arr(ages_perr[xsel])
        ages_merr = arr(ages_merr[xsel])
        ages_errs = np.array(
                [ages_perr, ages_merr]).reshape(2, len(ages[xsel]))
        yerrs = ages_errs

    else:
        xvals, yvals = arr(goodvals), arr(ages)
        ages_errs = np.array(
                [ages_perr, ages_merr]).reshape(2, len(ages))
        yerrs = ages_errs

    if is_cks and not logy:
        yvals /= 1e9
    if is_cks and logy:
        yerrs *= 1e9

    plt.close('all')
    f, ax = plt.subplots(figsize=(8,6))

    # squares
    ax.errorbar(xvals, yvals,
                elinewidth=0, ecolor='k', capsize=0, capthick=0,
                linewidth=0, fmt='s', ms=3, zorder=1)
    # error bars
    ax.errorbar(xvals, yvals, yerr=yerrs,
                elinewidth=0.3, ecolor='k', capsize=0, capthick=0,
                linewidth=0, fmt='s', ms=0, zorder=2, alpha=0.05)

    if '_' in xparam:
        ax.set_xlabel(' '.join(xparam.split('_')))
    else:
        ax.set_xlabel(xparam)
    if is_cks and 'giso_prad' in xparam:
        ax.set_xlabel('cks VII planet radius [Re]')
    if is_cks and 'koi_dor' in xparam:
        ax.set_xlabel('KOI a/Rstar')
    if is_exoarchive:
        ax.set_ylabel('age [gyr] (from exoarchive)')
    elif is_sandersdas:
        ax.set_ylabel('age [gyr] (from Sanders Das 2018)')
    elif is_cks:
        ax.set_ylabel('age [gyr] (from CKS VII)')

    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    if not logy:
        ax.set_ylim([0,14])

    f.tight_layout()

    logystr = 'log_' if logy else ''
    logxstr = 'log_' if logx else ''

    if is_exoarchive:
        savdir = '../results/exoarchive_age_plots{:s}/'.format(savdir_append)
    elif is_sandersdas:
        savdir = '../results/sd18_age_plots{:s}/'.format(savdir_append)
    elif is_cks:
        savdir = '../results/cks_age_plots{:s}/'.format(savdir_append)
    if not os.path.exists(savdir):
        os.mkdir(savdir)


    fname_pdf = logystr+'age_vs_'+logxstr+xparam+'.pdf'
    fname_png = fname_pdf.replace('.pdf','.png')
    f.savefig(savdir+fname_pdf)
    print('saved {:s}'.format(fname_pdf))
    f.savefig(savdir+fname_png, dpi=250)


def plot_jankyparam(tab, finite_age_inds, xparam, logx, logy,
                    is_exoarchive=False, is_cks=False, is_sandersdas=False,
                    outpdfpath=None, elw=0.3, ealpha=0.05, savdir_append=''):
    '''
    x axis: "xparam". y axis: age.

    args:
        tab (DataFrame or astropy table)
        finite_age_inds
        xparam (str): thing you want to scatter plot again age, with
            two-sided errors.

    kwargs:
        only one of is_exoarchive, is_cks, is_sandersdas should be true.

        outpdfpath: overrides save path

        elw: errorlinewidth
    '''

    # only one of is_exoarchive, is_cks, or is_sandersdas should be True.
    assert np.sum(np.array([(is_exoarchive != is_cks),
                  (is_cks != is_sandersdas),
                  (is_exoarchive != is_sandersdas)]).astype(int)) == 2

    if is_exoarchive:
        ages = tab['st_age'][finite_age_inds]
        ages_perr = tab['st_ageerr1'][finite_age_inds]
        ages_merr = np.abs(tab['st_ageerr2'][finite_age_inds])
    elif is_cks:
        ages = 10**(tab['giso_slogage'][finite_age_inds])
        ages_perr = 10**(tab['giso_slogage_err1'][finite_age_inds])
        ages_merr = 10**(np.abs(tab['giso_slogage_err2'][finite_age_inds]))
    elif is_sandersdas:
        ages = 10**(tab['log10_age'][finite_age_inds])
        ages_perr = 10**(tab['log10_age_err'][finite_age_inds])
        ages_merr = 10**(np.abs(tab['log10_age_err'][finite_age_inds]))
    else:
        raise NotImplementedError

    ages_errs = np.array(
            [ages_perr, ages_merr]).reshape(2, len(ages))

    goodvals = tab[xparam][finite_age_inds]

    if is_exoarchive or is_sandersdas:
        goodvals_perr = tab[xparam+'err1'][finite_age_inds]
        goodvals_merr = np.abs(tab[xparam+'err2'][finite_age_inds])
    elif is_cks:
        if 'smet' in xparam:
            goodvals_perr = tab['cks_smet_err1'][finite_age_inds]
            goodvals_merr = np.abs(tab['cks_smet_err2'][finite_age_inds])
        else:
            goodvals_perr = tab[xparam+'_err1'][finite_age_inds]
            goodvals_merr = np.abs(tab[xparam+'_err2'][finite_age_inds])
    else:
        raise NotImplementedError

    goodvals_errs = np.array(
        [goodvals_perr, goodvals_merr]).reshape(2, len(goodvals))

    plt.close('all')
    f, ax = plt.subplots(figsize=(8,6))

    if is_cks and not logy:
        ages /= 1e9
    if is_cks and logy:
        ages_errs *= 1e9

    # squares
    ax.errorbar(arr(goodvals), arr(ages),
                elinewidth=0, ecolor='k', capsize=0, capthick=0,
                linewidth=0, fmt='s', ms=3, zorder=2)
    # error bars
    ax.errorbar(arr(goodvals), arr(ages), yerr=ages_errs,
                xerr=goodvals_errs, elinewidth=elw, ecolor='k',
                capsize=0, capthick=0, linewidth=0, fmt='s', ms=0,
                zorder=1, alpha=ealpha)

    if '_' in xparam:
        ax.set_xlabel(' '.join(xparam.split('_')))
    else:
        ax.set_xlabel(xparam)

    if is_exoarchive:
        ax.set_ylabel('age [gyr] (from exoarchive)')
    elif is_sandersdas:
        ax.set_ylabel('age [gyr] (from Sanders Das 2018)')
    elif is_cks:
        ax.set_ylabel('age [gyr] (from CKS VII)')
    else:
        raise NotImplementedError


    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    if not logy:
        ax.set_ylim([0,14])
    if 'orbincl' in xparam:
        ax.set_xlim([75,93])
    if 'eccen' in xparam:
        ax.set_xlim([-0.05,1.05])

    f.tight_layout()

    logystr = 'log_' if logy else ''
    logxstr = 'log_' if logx else ''

    if is_exoarchive:
        savdir = '../results/exoarchive_age_plots{:s}/'.format(savdir_append)
    elif is_sandersdas:
        savdir = '../results/sd18_age_plots{:s}/'.format(savdir_append)
    elif is_cks:
        savdir = '../results/cks_age_plots{:s}/'.format(savdir_append)
    else:
        raise NotImplementedError
    if not os.path.exists(savdir):
        os.mkdir(savdir)

    fname_pdf = logystr+'age_vs_'+logxstr+xparam+'.pdf'
    fname_png = fname_pdf.replace('.pdf','.png')
    if outpdfpath:
        f.savefig(outpdfpath)
        f.savefig(outpdfpath.replace('.pdf','.png'))
        print('saved {:s}'.format(outpdfpath))
    else:
        f.savefig(savdir+fname_pdf)
        f.savefig(savdir+fname_png, dpi=250)
        print('saved {:s}'.format(fname_pdf))


def make_age_histograms_get_f_inds(df, is_cks=True, outstr='', logage=False,
                                   actually_make_plots=True,
                                   savdir='../results/cks_age_hist_precision/'):
    '''
    Saves to
        '../results/sigmaage_by_age_hist_cks{outstr}.pdf'
        '../results/age_hist_cks{outstr}.pdf'
    '''

    if not is_cks:
        raise NotImplementedError

    # replace 0's with nans.
    f_ages = np.isfinite(df['giso_slogage'])
    f_p_ages = np.isfinite(df['giso_slogage_err1'])
    f_m_ages = np.isfinite(df['giso_slogage_err2'])

    sel = f_ages & f_p_ages & f_m_ages
    if not actually_make_plots:
        return sel

    sigma_tau_cks = np.sqrt((10**(arr(df['giso_slogage_err1'])[sel]))**2 + \
                            (10**(np.abs(arr(df['giso_slogage_err2'])[sel])))**2)
    tau_cks = 10**(arr(df['giso_slogage'])[sel])
    log10_tau_cks = arr(df['giso_slogage'])[sel]
    log10_sigma_tau_cks = np.sqrt(arr(df['giso_slogage_err1'])[sel]**2 + \
                                  arr(df['giso_slogage_err2'])[sel]**2)

    sigma_tau_by_tau_cks = sigma_tau_cks / (tau_cks/1e9)

    ##########
    plt.close('all')
    plt.hist(sigma_tau_by_tau_cks[sigma_tau_by_tau_cks<=2.1], histtype='step', bins=20)
    plt.xlabel(r'$\sigma_{\mathrm{age}}/\mathrm{age}$, from CKS VII. '
               '$\sigma_{\mathrm{age}}$ from quadrature.')
    plt.xlim([0,2])
    plt.ylabel('count')
    plt.tight_layout()

    if not os.path.exists(savdir):
        os.mkdir(savdir)
    plt.savefig(savdir+'sigmaage_by_age_hist_cks{:s}.pdf'.format(outstr))

    plt.close('all')
    if not logage:
        plt.hist(tau_cks/1e9, histtype='step', bins=20)
        plt.errorbar(10, 100, yerr=0, xerr=np.median(sigma_tau_cks)/2, ecolor='k',
                     capsize=2, elinewidth=2, capthick=2)
        plt.errorbar(10, 80, yerr=0, xerr=np.mean(sigma_tau_cks)/2, ecolor='b',
                     capsize=2, elinewidth=2, capthick=2)
    else:
        plt.hist(np.log10(tau_cks), histtype='step',
                 bins=np.arange(8.5,10.25+.125,0.125))
    # divide by two because xerrs are plus and minus, and we want 1sigma, not
    # 2*(1sigma).
    if not logage:
        plt.xlabel('age from CKS VII [Gyr]. median error (top), mean error '
                   '(bottom).')
    else:
        plt.xlabel('log(age[Gyr]) from CKS VII. median error (top), mean error '
                   '(bottom).')
    plt.ylabel('count')
    #if logage:
    #    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(savdir+'age_hist_cks{:s}.pdf'.format(outstr))

    return sel


def make_stacked_histograms(df, xparam='radius', logtime=False,
                            savdir='../results/cks_age_explorn_stackedhist/'):

    if xparam=='radius':
        xlabel='giso_prad'
    elif xparam=='period':
        xlabel='koi_period'
    elif xparam=='aoverRstar':
        xlabel='koi_dor'
    else:
        raise NotImplementedError

    # log radius bins
    logbin_radii = np.logspace(-0.5, 1.5, num=21)
    logbin_period = np.logspace(-0.5, 2.25, num=12)
    logbin_aoverRstar = np.logspace(0, 2.25, num=10)
    if xparam=='radius':
        logbin_x = logbin_radii
    elif xparam=='period':
        logbin_x = logbin_period
    elif xparam=='aoverRstar':
        logbin_x = logbin_aoverRstar
    else:
        raise NotImplementedError

    # linear age bins, 1-gyr spaced.
    linbins = np.arange(0, 1.5e10, 1e9)
    left_linbins, right_linbins = linbins[:-1], linbins[1:]
    linbin_intervals = list(zip(left_linbins, right_linbins))
    # log age bins, in quarter-dex from 1e9 to 1e10
    logbins = np.logspace(8.75, 10.25, 7)
    left_logbins, right_logbins = logbins[:-1], logbins[1:]
    logbin_intervals = list(zip(left_logbins, right_logbins))

    if logtime:
        age_intervals = logbin_intervals
    else:
        age_intervals = linbin_intervals

    ages = 10**arr(df['giso_slogage'])

    plt.close('all')
    f,axs = plt.subplots(nrows=len(age_intervals), sharex=True, figsize=(8,12))

    for ix, age_interval in enumerate(age_intervals):

        minage, maxage = min(age_interval), max(age_interval)

        print(ix)
        ax = axs[ix]
        ax.set_xscale('log')

        sel = (ages > minage) & (ages < maxage)
        sel &= np.isfinite(arr(df[xlabel]))
        if xparam=='aoverRstar':
            sel &= arr(df[xlabel]) < 177.8

        N_cks = len(df[xlabel][sel])

        try:
            weights = np.ones_like(df[xlabel][sel])/float(N_cks)

            # be sure to plot the fraction of planets in each bin. (not the
            # pdf, which has INTEGRAL normalized to 1).
            hist, _, _ = ax.hist(df[xlabel][sel], bins=logbin_x,
                                 histtype='stepfilled', weights=weights)
        except:
            print('caught err in ax.hist, getting out')

        if logtime:
            agestr = '{:.2f}<log(age)<{:.2f}\nN={:d}'.format(
                     np.log10(minage), np.log10(maxage), int(N_cks))
        else:
            agestr = '{:.1f}<age(Gyr)<{:.1f}\nN={:d}'.format(
                     minage/1e9,maxage/1e9, int(N_cks))

        if xparam=='radius':
            text_yval = 0.17
            text_xval = 30
        elif xparam=='period':
            text_yval = 0.17
            text_xval = 200
        elif xparam=='aoverRstar':
            text_yval = 0.17
            text_xval = 170
        else:
            raise NotImplementedError

        ax.text(text_xval, text_yval, agestr,
                horizontalalignment='right',
                verticalalignment='top',
                fontsize='xx-small')

        if logtime and xparam=='radius':
            ax.set_ylim([0,0.2])
        elif not logtime and xparam=='radius':
            ax.set_ylim([0,0.2])
        elif logtime and xparam=='period':
            ax.set_ylim([0,0.2])
        elif not logtime and xparam=='period':
            ax.set_ylim([0,0.2])
        elif logtime and xparam=='aoverRstar':
            ax.set_ylim([0,0.2])
        elif not logtime and xparam=='aoverRstar':
            ax.set_ylim([0,0.2])

        if ix in [5,6,7,8] and not logtime:
            ax.set_yticklabels('')
        if ix in [2,3] and logtime:
            ax.set_yticklabels('')

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

    if xparam=='radius':
        ax.set_xlabel('CKS VII planet radius [Re]')
    elif xparam=='period':
        ax.set_xlabel('KOI period [d] (less than 180d)')
    elif xparam=='aoverRstar':
        ax.set_xlabel('a/Rstar (less than 178)')
    else:
        raise NotImplementedError

    f.text(0.02, 0.5, 'fraction of planets in each age bin', ha='left',
           va='center', rotation='vertical', fontsize='large')

    f.tight_layout(h_pad=-0.9)

    if not os.path.exists(savdir):
        os.mkdir(savdir)

    if logtime:
        savpath = savdir+'stackedhist_log_time_vs_{:s}.pdf'.format(xparam)
    else:
        savpath = savdir+'stackedhist_linear_time_vs_{:s}.pdf'.format(
                  xparam)
    f.savefig(savpath, bbox_inches='tight')
    f.savefig(savpath.replace('.pdf','.png'), dpi=300, bbox_inches='tight')

    print('made {:s}'.format(savpath))


def make_quartile_scatter(df, xparam='koi_period',
                          savdir='../results/cks_age_scatter_percentiles/'):
    '''
    Reproduce Petigura+ 2018's Figure 4, but using ages instead of
    metallicities.

    I chose the percentiles here to match the fraction of planets in each bin
    given by Petigura+ 2018 Fig 4.

    This is somewhat wrong! However I couldn't think of anything smarter. The
    idea is that each age bin should contain an equal fraction (25%) of the
    parent stellar sample, S.  However, there is no parent stellar sample with
    good ages.  Petigura+ 2018's approach to the analogous problem with
    metallicities was to use LAMOST DR3. However, LAMOST produces Teff, logg,
    and [Fe/H]. No isochrone+spectral metallicity ages.  (This might be a
    problem worth attacking in itself).

    One approach might be to use the Sanders & Das 2018 ages, cross-matched to
    Mathur+ 2017's DR25 Kepler target list. The above ages do take into account
    the spectroscopic metallicities. I'm not convinced that it's a good idea
    though.

    kwargs:
        xparam: 'koi_period' or 'koi_dor' (a/Rstar)

    '''
    if not os.path.exists(savdir):
        os.mkdir(savdir)

    ages = 10**arr(df['giso_slogage'])

    # to match Petigura's Fig 4.
    younger, median, older = 28, 58, 80
    ages_younger_25th = np.percentile(ages, younger)
    ages_median = np.percentile(ages, median)
    ages_older_75th = np.percentile(ages, older)

    ages_percentiles = [ages_younger_25th, ages_median, ages_older_75th]

    plt.close('all')

    f,axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(6,6))

    axs = axs.flatten()

    for ix, ax in enumerate(axs):

        ax.scatter(arr(df[xparam]), arr(df['giso_prad']),
                   zorder=1, marker='o', s=3, c='lightgray')

        if ix == 0:
            sel = ages < ages_percentiles[ix]
            textstr = 'age$<${:.2f}Gyr\nNtot={:d},Nblue={:d}\nfp={:.2f}'.format(
                ages_percentiles[ix]/1e9,len(df),len(df[sel]),len(df[sel])/len(df))
        elif ix == len(axs)-1:
            sel = ages > ages_percentiles[ix-1]
            textstr = 'age$>${:.2f}Gyr\nNtot={:d},Nblue={:d}\nfp={:.2f}'.format(
                ages_percentiles[ix-1]/1e9,len(df),len(df[sel]),len(df[sel])/len(df))
        else:
            sel = ages >= ages_percentiles[ix-1]
            sel &= ages < ages_percentiles[ix]
            textstr = 'age=({:.2f},{:.2f})Gyr\nNtot={:d},Nblue={:d}\nfp={:.2f}'.format(
                ages_percentiles[ix-1]/1e9,ages_percentiles[ix]/1e9,len(df),len(df[sel]),len(df[sel])/len(df))

        ax.scatter(arr(df[xparam])[sel], arr(df['giso_prad'])[sel],
                   marker='o', s=3, c='#1f77b4', zorder=2)

        ax.text(0.99, 0.99, textstr, horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                fontsize='xx-small')

        meanmetbin = np.mean(arr(df['cks_smet'])[sel])
        stdmetbin = np.std(arr(df['cks_smet'])[sel])
        meanmetstr = 'mean([Fe/H])={:.2f}\nstd([Fe/H])={:.2f}'.format(
                     meanmetbin, stdmetbin )
        ax.text(0.99, 0.01, meanmetstr, horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes,
                fontsize='xx-small')

        ax.set_xscale('log')
        ax.set_yscale('log')
        if xparam=='koi_period':
            ax.set_xlim([0.3,330])
        elif xparam=='koi_dor':
            ax.set_xlim([1,200])
        else:
            raise NotImplementedError
        ax.set_ylim([0.5,32])

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

        del sel

    f.tight_layout(h_pad=0, w_pad=0)

    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)
    plt.grid(False)
    if xparam=='koi_period':
        plt.xlabel("orbital period [days]")
        savpath = savdir+'rp_vs_period_scatter_quartiles.pdf'
    elif xparam=='koi_dor':
        plt.xlabel("a/Rstar")
        savpath = savdir+'rp_vs_aoverRstar_scatter_quartiles.pdf'
    plt.ylabel("planet radius [earth radii]")

    f.savefig(savpath, bbox_inches='tight')
    f.savefig(savpath.replace('.pdf','.png'), dpi=300, bbox_inches='tight')

    print('made {:s}'.format(savpath))


def make_boxplot_koi_count(df):
    rc('text', usetex=False)
    import seaborn as sns
    sns.set(style='ticks')
    plt.close('all')
    f,ax = plt.subplots(figsize=(8,6))

    df['giso_sage'] = 10**arr(df['giso_slogage'])

    sns.boxplot(x='koi_count', y='giso_sage', data=df, whis=0., palette='vlag')

    sns.swarmplot(x='koi_count', y='giso_sage', data=df, size=2, color='.3',
                  linewidth=0)

    f.tight_layout()

    savdir = '../results/cks_age_plots/'
    fname_pdf = 'age_vs_koi_count_boxplot.pdf'
    fname_png = fname_pdf.replace('.pdf','.png')
    f.savefig(savdir+fname_pdf)
    print('saved {:s}'.format(fname_pdf))
    f.savefig(savdir+fname_png, dpi=250)


def make_octile_scatter(df, xparam='koi_period',
                        savdir='../results/cks_age_scatter_percentiles/'):
    '''
    Reproduce Petigura+ 2018's Figure 4, but using ages instead of
    metallicities.

    I chose the percentiles here as just 1/8th the age distribution of detected
    planets.

    Another approach might be to use the Sanders & Das 2018 ages, cross-matched
    to Mathur+ 2017's DR25 Kepler target list. The above ages do take into
    account the spectroscopic metallicities. I'm not convinced that it's a good
    idea though.

    kwargs:
        xparam: 'koi_period' or 'koi_dor' (a/Rstar)

    '''

    ages = 10**arr(df['giso_slogage'])

    octiles = np.arange(0,100,12.5)
    ages_octiles = np.array([np.percentile(ages, octile) for octile in octiles])

    plt.close('all')

    f,axs = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True,
                         figsize=(12,6))

    axs = axs.flatten()

    for ix, ax in enumerate(axs):

        ax.scatter(arr(df[xparam]), arr(df['giso_prad']),
                   zorder=1, marker='o', s=3, c='lightgray')

        if ix == 0:
            sel = ages < ages_octiles[ix+1]
            textstr = 'age $<$ {:.2f}Gyr\nfp=12.5%'.format(
                ages_octiles[ix]/1e9)
        elif ix == 7:
            sel = ages > ages_octiles[ix]
            textstr = 'age $>$ {:.2f}Gyr\nfp=12.5%'.format(
                ages_octiles[ix]/1e9)
        else:
            sel = ages >= ages_octiles[ix]
            sel &= ages < ages_octiles[ix+1]
            textstr = 'age = ({:.2f},{:.2f})Gyr\nfp=12.5%'.format(
                ages_octiles[ix]/1e9,ages_octiles[ix+1]/1e9)

        ax.scatter(arr(df[xparam])[sel], arr(df['giso_prad'])[sel],
                   marker='o', s=3, c='#1f77b4', zorder=2)

        ax.text(0.95, 0.95, textstr, horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                fontsize='x-small')

        #ax.text(0.5, 0.5, repr(ix), horizontalalignment='right',
        #        verticalalignment='top', transform=ax.transAxes)

        ax.set_xscale('log')
        ax.set_yscale('log')
        if xparam=='koi_period':
            ax.set_xlim([0.3,330])
        elif xparam=='koi_dor':
            ax.set_xlim([1,200])
        else:
            raise NotImplementedError
        ax.set_ylim([0.5,32])

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

        del sel

    f.tight_layout(h_pad=0, w_pad=0)

    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)
    plt.grid(False)

    if not os.path.exists(savdir):
        os.mkdir(savdir)
    if xparam=='koi_period':
        plt.xlabel("orbital period [days]")
        savpath = savdir+'rp_vs_period_scatter_octiles.pdf'
    elif xparam=='koi_dor':
        plt.xlabel("a/Rstar")
        savpath = savdir+'rp_vs_aoverRstar_scatter_octiles.pdf'
    plt.ylabel("planet radius [earth radii]")

    f.savefig(savpath, bbox_inches='tight')
    f.savefig(savpath.replace('.pdf','.png'), dpi=300, bbox_inches='tight')

    print('made {:s}'.format(savpath))


def make_old_young_scatter(df, xparam='koi_period', metlow=None, methigh=None,
                          logagehigh=None, logagelow=None,
                          savdir='../results/cks_age_scatter_percentiles/'):
    '''
    option 1: set methigh and metlow.
        Then this fixes the metallicity. And scatter plots Rp vs Porb,
        percentiling by AGE.
    option 2: set logagehigh and logagelow.
        Then this fixes the age. And scatter plots Rp vs Porb, percentiling by
        METALLICITY.
    '''

    assert not (type(methigh)==float) & (type(logagehigh)==float)
    if type(methigh)==float:
        metstr = 'cut on [Fe/H]=[{:.2f},{:.2f})'.format(metlow,methigh)
        print(metstr)
        subsel = arr(df['cks_smet']) < methigh
        subsel &= arr(df['cks_smet']) >= metlow
    elif type(logagehigh)==float:
        logagestr = 'cut on logage=[{:.2f},{:.2f})'.format(logagelow,logagehigh)
        print(logagestr)
        subsel = arr(df['giso_slogage']) < logagehigh
        subsel &= arr(df['giso_slogage']) >= logagelow
    else:
        raise NotImplementedError

    df = df[subsel]

    ages = 10**arr(df['giso_slogage'])
    mets = arr(df['cks_smet'])

    #sep = 33.3333333
    sep = 25
    percentiles = np.arange(0,100,sep)
    if type(methigh)==float:
        sel_percentiles = np.array([np.percentile(ages, percentile) for percentile in percentiles])
    elif type(logagehigh)==float:
        sel_percentiles = np.array([np.percentile(mets, percentile) for percentile in percentiles])

    plt.close('all')

    f,axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True,
                         figsize=(16,4))

    axs = axs.flatten()

    for ix, ax in enumerate(axs):

        ax.scatter(arr(df[xparam]), arr(df['giso_prad']),
                   zorder=1, marker='o', s=3, c='lightgray')

        if type(methigh)==float:
            if ix == 0:
                sel = ages < sel_percentiles[ix+1]
                textstr = 'age$<${:.2f}Gyr\nNtot={:d},Nblue={:d}\n{:s}'.format(
                    sel_percentiles[ix+1]/1e9,len(df),int(len(df)*sep/100),metstr)
            elif ix == len(axs)-1:
                sel = ages > sel_percentiles[ix]
                textstr = 'age$>${:.2f}Gyr\nNtot={:d},Nblue={:d}\n{:s}'.format(
                    sel_percentiles[ix]/1e9,len(df),int(len(df)*sep/100),metstr)
            else:
                sel = ages >= sel_percentiles[ix]
                sel &= ages < sel_percentiles[ix+1]
                textstr = 'age=({:.2f},{:.2f})Gyr\nNtot={:d},Nblue={:d}\n{:s}'.format(
                    sel_percentiles[ix]/1e9,sel_percentiles[ix+1]/1e9,len(df),int(len(df)*sep/100),metstr)

        elif type(logagehigh)==float:
            if ix == 0:
                sel = mets < sel_percentiles[ix+1]
                textstr = 'Fe/H$<${:.2f}\nNtot={:d},Nblue={:d}\n{:s}'.format(
                    sel_percentiles[ix+1],len(df),int(len(df)*sep/100),logagestr)
            elif ix == len(axs)-1:
                sel = mets > sel_percentiles[ix]
                textstr = 'Fe/H$>${:.2f}\nNtot={:d},Nblue={:d}\n{:s}'.format(
                    sel_percentiles[ix],len(df),int(len(df)*sep/100),logagestr)
            else:
                sel = mets >= sel_percentiles[ix]
                sel &= mets < sel_percentiles[ix+1]
                textstr = 'Fe/H=({:.2f},{:.2f})\nNtot={:d},Nblue={:d}\n{:s}'.format(
                    sel_percentiles[ix],sel_percentiles[ix+1],len(df),int(len(df)*sep/100),logagestr)

        ax.scatter(arr(df[xparam])[sel], arr(df['giso_prad'])[sel],
                   marker='o', s=3, c='#1f77b4', zorder=2)

        ax.text(0.95, 0.95, textstr, horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes,
                fontsize='xx-small')

        if type(methigh)==float:
            meanmetbin = np.mean(arr(df['cks_smet'])[sel])
            stdmetbin = np.std(arr(df['cks_smet'])[sel])
            meanmetstr = 'mean([Fe/H])={:.2f}\nstd([Fe/H])={:.2f}'.format(
                         meanmetbin, stdmetbin )
            ax.text(0.95, 0.04, meanmetstr, horizontalalignment='right',
                    verticalalignment='bottom', transform=ax.transAxes,
                    fontsize='xx-small')

        ax.set_xscale('log')
        ax.set_yscale('log')
        if xparam=='koi_period':
            ax.set_xlim([0.3,330])
        elif xparam=='koi_dor':
            ax.set_xlim([1,200])
        else:
            raise NotImplementedError
        ax.set_ylim([0.5,32])

        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')

        del sel

    f.tight_layout(h_pad=0, w_pad=0)

    f.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)
    plt.grid(False)

    if not os.path.exists(savdir):
        os.mkdir(savdir)
    if type(methigh)==float:
        if xparam=='koi_period':
            plt.xlabel("orbital period [days]")
            savpath = savdir+'rp_vs_period_'+\
                      'metlow{:.2f}_methigh{:.2f}_scatter_percentiles.pdf'.format(
                      metlow,methigh)
        elif xparam=='koi_dor':
            plt.xlabel("a/Rstar")
            savpath = savdir+'rp_vs_aoverRstar_'+\
                      'metlow{:.2f}_methigh{:.2f}_scatter_percentiles.pdf'.format(
                      metlow,methigh)

    elif type(logagehigh)==float:
        if xparam=='koi_period':
            plt.xlabel("orbital period [days]")
            savpath = savdir+'rp_vs_period_'+\
                      'logagelow{:.2f}_logagehigh{:.2f}_scatter_percentiles.pdf'.format(
                      logagelow,logagehigh)
        elif xparam=='koi_dor':
            plt.xlabel("a/Rstar")
            savpath = savdir+'rp_vs_aoverRstar_'+\
                      'logagelow{:.2f}_logagehigh{:.2f}_scatter_percentiles.pdf'.format(
                      logagelow,logagehigh)

    plt.ylabel("planet radius [earth radii]")

    f.savefig(savpath, bbox_inches='tight')
    f.savefig(savpath.replace('.pdf','.png'), dpi=300, bbox_inches='tight')

    print('made {:s}'.format(savpath))


def plot_scatter(tab, finite_age_inds, xparam, yparam, logx, logy,
                 is_cks=True, savdir='../results/cks_scatter_plots/',
                 ylim=None, xlim=None):
    '''
    args:
        tab (DataFrame or astropy table)
        finite_age_inds
        yparam (str): preferably 'giso_slogage' or 'cks_smet'
        xparam (str): thing you want to scatter plot against
        xlim (tuple): overrides

    kwargs:
        only one of is_exoarchive, is_cks, is_sandersdas should be true.
    '''

    goodvals = tab[xparam][finite_age_inds]

    if is_cks:
        if yparam=='giso_slogage':
            yvals = 10**(tab['giso_slogage'][finite_age_inds])
            yvals_perr = 10**(tab['giso_slogage_err1'][finite_age_inds])
            yvals_merr = 10**(np.abs(tab['giso_slogage_err2'][finite_age_inds]))
        elif yparam=='cks_smet':
            yvals = tab[yparam][finite_age_inds]
            yvals_perr = tab[yparam][finite_age_inds]
            yvals_merr = tab[yparam][finite_age_inds]
        if xparam=='giso_slogage':
            xvals = 10**(tab['giso_slogage'][finite_age_inds])
            xvals_perr = 10**(tab['giso_slogage_err1'][finite_age_inds])
            xvals_merr = 10**(np.abs(tab['giso_slogage_err2'][finite_age_inds]))
        elif xparam=='cks_smet':
            xvals = tab[yparam][finite_age_inds]
            xvals_perr = tab[yparam][finite_age_inds]
            xvals_merr = tab[yparam][finite_age_inds]
    else:
        raise NotImplementedError

    xvals, yvals = arr(goodvals), arr(yvals)
    yvals_errs = np.array(
            [yvals_perr, yvals_merr]).reshape(2, len(yvals))
    yerrs = yvals_errs

    if is_cks and not logy and 'age' in yparam:
        yvals /= 1e9
    if is_cks and logy and 'age' in yparam:
        yerrs *= 1e9

    plt.close('all')
    f, ax = plt.subplots(figsize=(8,6))

    # squares
    ax.errorbar(xvals, yvals,
                elinewidth=0, ecolor='k', capsize=0, capthick=0,
                linewidth=0, fmt='s', ms=3, zorder=1)
    # error bars
    ax.errorbar(xvals, yvals, yerr=yerrs,
                elinewidth=0.3, ecolor='k', capsize=0, capthick=0,
                linewidth=0, fmt='s', ms=0, zorder=2, alpha=0.05)

    if '_' in xparam:
        ax.set_xlabel(' '.join(xparam.split('_')))
    else:
        ax.set_xlabel(xparam)
    if is_cks and 'giso_prad' in xparam:
        ax.set_xlabel('cks VII planet radius [Re]')
    if is_cks and 'koi_dor' in xparam:
        ax.set_xlabel('KOI a/Rstar')
    if is_cks:
        ax.set_ylabel(yparam.replace('_',' '))

    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    if ylim is None and not logy and 'age' in yparam:
        ax.set_ylim([0,14])
    elif not logy and 'smet' in yparam and 'prad' in xparam:
        ax.set_ylim([-0.2,0.4])
        ax.set_xlim([0.3,20])

    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    f.tight_layout()

    logystr = 'log_' if logy else ''
    logxstr = 'log_' if logx else ''

    if not os.path.exists(savdir):
        os.mkdir(savdir)

    fname_pdf = logystr+yparam+'_vs_'+logxstr+xparam+'.pdf'
    fname_png = fname_pdf.replace('.pdf','.png')
    f.savefig(savdir+fname_pdf)
    print('saved {:s}'.format(fname_pdf))
    f.savefig(savdir+fname_png, dpi=250)


def plot_hr(tab, finite_age_inds, colorkey, is_cks=True,
            savdir='../results/cks_scatter_color_plots/', ylim=None,
            xlim=None):
    ''' plot a HR diagram with colored points.

    args:
        tab (DataFrame or astropy table)
        finite_age_inds
        colorkey (str): preferably 'giso_slogage' or 'cks_smet'
        xlim (tuple): overrides

    kwargs:
        only one of is_exoarchive, is_cks, is_sandersdas should be true.
    '''

    if is_cks:
        xvals = arr(tab['cks_steff'][finite_age_inds])
        dist_pc = 1/( arr(tab['gaia2_sparallax']*1e-3)[finite_age_inds] )
        mu = 5 * np.log10(dist_pc)  - 5
        kmag_apparent = arr(tab['m17_kmag'])[finite_age_inds]
        kmag_absolute = kmag_apparent - mu
        yvals = kmag_absolute
        colors = 10**(arr( tab['giso_slogage'][finite_age_inds]) ) / 1e9
    else:
        raise NotImplementedError

    plt.close('all')
    f, ax = plt.subplots(figsize=(8,6))

    import seaborn as sns

    bounds= list(np.arange(0.5,15.5,1)) #[0, 5, 10]
    ncolor = len(bounds)

    cmap1 = mpl.colors.ListedColormap(
            sns.color_palette("plasma", n_colors=ncolor, desat=1))
    norm1 = mpl.colors.BoundaryNorm(bounds, cmap1.N)

    # squares
    out = ax.scatter(xvals, yvals, marker='s', c=colors, s=8,
                     zorder=1, cmap=cmap1, norm=norm1, rasterized=True)

    cbar = f.colorbar(out, cmap=cmap1, norm=norm1, boundaries=bounds,
        fraction=0.025, pad=0.02, ticks=np.arange(ncolor)+1,
        orientation='vertical')

    cbarlabels = list(map(str,arr(arr(bounds)+0.5).astype(int)))
    cbar.ax.set_yticklabels(cbarlabels)
    cbar.set_label('gaia+cks isochrone age [Gyr]', rotation=270, labelpad=7)

    ax.set_xlabel('cks teff [K]')
    ax.set_ylabel('absolute K mag (Mathur17 + GaiaDR2 d=1/plx)')

    ax.set_xscale('log')

    xlim = ax.get_xlim()
    ax.set_xlim(max(xlim),min(xlim))
    ylim = ax.get_ylim()
    ax.set_ylim(max(ylim),min(ylim))

    f.tight_layout()

    if not os.path.exists(savdir):
        os.mkdir(savdir)

    fname_pdf = 'abskmag_vs_logteff.pdf'
    fname_png = fname_pdf.replace('.pdf','.png')
    f.savefig(savdir+fname_pdf)
    print('saved {:s}'.format(fname_pdf))
    f.savefig(savdir+fname_png, dpi=250)
