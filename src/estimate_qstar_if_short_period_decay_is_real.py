'''
if the dearth of the close-in planets at old ages is because of tidal decay,
this would imply that by ~5 Gyr, tidal decay -> disruption was starting to
happen for planets of LOW MASS.

what would the OOM constraint on Q be?
'''
from __future__ import division

import numpy as np
from astropy import constants as const, units as u


a_over_rstar = 4    # guesstimate from age vs a/Rstar CKS plot
t_evoln = 10*u.Gyr  # guesstimate from age vs a/Rstar CKS plot
period = 1*u.day    # guesstimate from age vs period CKS plot

M_p = 1*u.Mearth    # maybe 10?
M_star = 1*u.Msun # FGK stars in the CKS sample

n = 2*np.pi/period

R_star_by_a = 1/a_over_rstar

for M_p in [1*u.Mearth, 2*u.Mearth, 10*u.Mearth]:

    # e.g., Hellier et al 2009, who cite Eq 29 of Dobbs-Dixon, Lin & Mardling
    # (2004), and Eq 5 of Levrard et al 2009.
    Q = 117/2 * t_evoln * n * (M_p/M_star) * (R_star_by_a)**5

    print('\nassuming a/Rstar={:.1f}, t_inspiral={:.1f}, period={:.1f}'.format(
           a_over_rstar, t_evoln, period)+
        '\nM_p={:.1f}, M_star={:.1f}, and equilibrium tides,'.format(
        M_p, M_star)
    )

    print('\t Q={:.1e}'.format(Q.cgs))
