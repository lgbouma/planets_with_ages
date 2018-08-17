# -*- coding: utf-8 -*-
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
import pandas as pd, numpy as np

##########

def arr(x):
    return np.array(x)

##########

df = pd.read_csv('../data/ea_sd18_matches.csv')

# replace 0's with nans.
df = df.replace({'st_age':0., 'st_ageerr1':0., 'st_ageerr2':0.}, value=np.nan)

f_ages = np.isfinite(df['st_age'])
f_p_ages = np.isfinite(df['st_ageerr1'])
f_m_ages = np.isfinite(df['st_ageerr2'])

sel = f_ages & f_p_ages & f_m_ages

# plot sigma_tau/tau for exoarchive planets. NB st_age is in Gyr.
sigma_tau_ea = np.sqrt(arr(df['st_ageerr1'])[sel]**2 +
                       arr(df['st_ageerr2'])[sel]**2)
tau_ea = arr(df['st_age'])[sel]
sigma_tau_by_tau_ea = sigma_tau_ea / tau_ea

plt.close('all')
plt.hist(sigma_tau_by_tau_ea, histtype='step', bins=20)
plt.xlabel(r'$\sigma_\tau/\tau$, from exoarchive. $\sigma_\tau$ from '
           'quadrature.')
plt.xlim([0,2])
plt.ylabel('count')
plt.tight_layout()
plt.savefig('../results/sigmatau_by_tau_hist_exoarchive.pdf')

plt.close('all')
plt.hist(tau_ea, histtype='step', bins=20)
plt.xlabel(r'$\tau$, from exoarchive [Gyr]. nb errors are big.')
plt.ylabel('count')
plt.tight_layout()
plt.savefig('../results/tau_hist_exoarchive.pdf')

##########

# plot sigma_tau/tau for exoarchive planets, but using Sanders & Dal 2018 ages.

f_ages = np.isfinite(df['log10_age'])
f_errs_ages = np.isfinite(df['log10_age_err'])

sel = f_ages & f_errs_ages

sigma_tau_sd18 = 10**(arr(df['log10_age_err'])[sel])
tau_sd18 = 10**(arr(df['log10_age'])[sel])
log10_tau_sd18 = arr(df['log10_age'])[sel]
log10_sigma_tau_sd18 = arr(df['log10_age_err'])[sel]
flag_sd18 = arr(df['flag'])[sel]

sigma_tau_by_tau_sd18 = sigma_tau_sd18 / tau_sd18
log10_sigma_tau_by_log10_tau_sd18 = log10_sigma_tau_sd18 / log10_tau_sd18

sel_rel_tau_error = (sigma_tau_by_tau_sd18<2)

plt.close('all')
plt.hist(sigma_tau_by_tau_sd18[sel_rel_tau_error], histtype='step', bins=20)
plt.xlabel(r'$\sigma_\tau/\tau$, from SD18.')
plt.ylabel('count')
plt.xlim([0,2])
plt.tight_layout()
outname = '../results/sigmatau_by_tau_hist_sd18_withbadflags.pdf'
plt.savefig(outname)
print('saved {:s}'.format(outname))

plt.close('all')
plt.hist(tau_sd18[sel_rel_tau_error], histtype='step', bins=20)
plt.xlabel(r'$\tau$, from SD18 [Gyr]. errors are big, but formally less '
           'than exoarchive.')
plt.ylabel('count')
plt.tight_layout()
outname = '../results/tau_hist_sd18_withbadflags.pdf'
plt.savefig(outname)
print('saved {:s}'.format(outname))

# flag: (0 if good, 1 if bad isochrones, 2 if bad spectra, 3 if bad photometry,
# 4 if bad astrometry, 5 if bad mass, 6 if bad uncertainties, 7 if probable
# binaries)

sel_flag = (flag_sd18 == 0)

sel_sd18 = sel_flag & sel_rel_tau_error

plt.close('all')
plt.hist(sigma_tau_by_tau_sd18[sel_sd18], histtype='step', bins=20)
plt.xlabel(r'$\sigma_\tau/\tau$, from SD18.')
plt.ylabel('count')
plt.xlim([0,2])
plt.tight_layout()
outname = '../results/sigmatau_by_tau_hist_sd18.pdf'
plt.savefig(outname)
print('saved {:s}'.format(outname))

plt.close('all')
plt.hist(tau_sd18[sel_sd18], histtype='step', bins=20)
plt.xlabel(r'$\tau$, from SD18 [Gyr]. errors are big, but formally less '
           'than exoarchive.')
plt.ylabel('count')
plt.tight_layout()
outname = '../results/tau_hist_sd18.pdf'
plt.savefig(outname)
print('saved {:s}'.format(outname))




##########

# same deal as exoarchive_age_plots
from exoarchive_age_plots import plot_wellmeasuredparam, plot_jankyparam
make_wellmeasured = True
make_janky = True

wellmeasuredparams = ['pl_radj', 'pl_rade', 'pl_orbper', 'pl_pnum', 'st_glat']
jankyparams = ['pl_dens', 'pl_orbeccen', 'pl_orbincl']

rc('text', usetex=False)

for logy in [True, False]:
    for logx in [True, False]:
        for wellmeasuredparam in wellmeasuredparams:

            if not make_wellmeasured:
                continue

            plot_wellmeasuredparam(df[f_ages & f_errs_ages], sel_sd18,
                                   wellmeasuredparam, logx, logy,
                                   is_exoarchive=False)

# now do the same thing with "janky" measurements like inclination and
# eccentricity. show both error bars
for logy in [True, False]:
    for logx in [True, False]:
        for jankyparam in jankyparams:

            if not make_janky:
                continue

            plot_jankyparam(df[f_ages & f_errs_ages], sel_sd18, jankyparam,
                            logx, logy, is_exoarchive=False)




#TODO: look at things that were flagged by Sanders as odd, for whatever reason
#      (esp. binaries)

#TODO: did Fulton and Petigura DR2 + CKS include ages? 
