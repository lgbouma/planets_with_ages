'''
not age related: do short period planets with a/Rstar<5 have systematically
higher obliquities? here, we check against Winn 2017's sample and see there
aren't enough with good vsini measurements.
'''
from __future__ import division, print_function

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

df = pd.read_csv('../data/Winn_2017_ajaa93e3t1_ascii.txt', comment='#',
                 delimiter='\t')

df.rename(index=str, columns={'$v\sin i$':'vsini',
                              '${T}_{\mathrm{eff}}$':'Teff'}, inplace=True)

# ok. Winn's table only gives 54 stars, but his paper talks about 156. what's
# up? Ah -- these are all the stars with v > 4km/s, which were selected because
# of the large fractional uncertainties otherwise.

from cks_age_exploration import _get_cks_data
cks = _get_cks_data()

d = pd.merge(df, cks, how='left', left_on='KIC', right_on='id_kic')

# 84 planets orbiting the 54 stars.
# 3 of the planets have a/Rstar < 5
# 17 of them have a/Rstar < 10.

# NOTE: it's somewhat shocking that the state of the rotation period literature
# is that there are only 54 STARS with reliable Prot & vsini, with v > 4km/s.
# WTF...
