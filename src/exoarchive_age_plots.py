'''
DESCRIPTION
----------
Download exoplanet archive and make scatter plots.

* R_p vs age, density vs age
    -> are hot jupiters typically around younger stars?
    -> can we see photoevaporation in time? [less dense planets lose
       atmospheres, get smaller and more dense]
    -> do we see the radius gap move in time?
* period vs age, also semimaj vs age
    -> do we get more USPs around older stars?
    -> do we see evidence that hot jupiters are at shorter periods around older
       stars? (tidal decay)
* eccentricity vs age
    -> do we see eccentricity decreasing for older systems?
* number of planets in system vs age
    -> do we see fewer planets in older systems, because of stellar fly-bys
       exciting mutual inclinations?
* galactic latitude of detected planet system vs age
    -> stars near the plane are somewhat younger. seen in planets?


USAGE
----------
select desired plots from bools below. then:

$ python exoarchive_age_plots.py

'''
make_wellmeasured = True
make_janky = True
make_timesteps = False

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

import os

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

from plot_age_scatter import plot_wellmeasuredparam, plot_jankyparam

def arr(x):
    return np.array(x)


if __name__=='__main__':

    # columns described at
    # https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html
    ea_tab = NasaExoplanetArchive.get_confirmed_planets_table(all_columns=True,
                                                              show_progress=True)

    # radius in jupiter (earth) radii, period, number of planets in system,
    # galacitc latitude.
    wellmeasuredparams = ['pl_radj', 'pl_rade', 'pl_orbper', 'pl_pnum', 'st_glat']
    # janky measurements where errors matter: age in Gyr, density(g/cc), eccen,
    # inclination
    jankyparams = ['pl_dens', 'pl_orbeccen', 'pl_orbincl']

    # get systems with finite ages (has a value, and +/- error bar)
    has_age_value = ~ea_tab['st_age'].mask
    has_age_errs  = (~ea_tab['st_ageerr1'].mask) & (~ea_tab['st_ageerr2'].mask)
    finite_age_inds = has_age_value & has_age_errs

    # plot age vs all "good cols". (age is on y axis b/c it has the error bars, and
    # I at least skimmed the footnotes of Hogg 2010)

    for logy in [True, False]:
        for logx in [True, False]:
            for wellmeasuredparam in wellmeasuredparams:

                if not make_wellmeasured:
                    continue

                plot_wellmeasuredparam(ea_tab, finite_age_inds,
                                       wellmeasuredparam, logx, logy)

    # now do the same thing with "janky" measurements like inclination and
    # eccentricity. show both error bars
    for logy in [True, False]:
        for logx in [True, False]:
            for jankyparam in jankyparams:

                if not make_janky:
                    continue

                plot_jankyparam(ea_tab, finite_age_inds, jankyparam, logx, logy)

    # Rp vs Porb, stepping thru time.
    if make_timesteps:
        raise NotImplementedError
