'''
DESCRIPTION
----------

Plotting script to make the skymap figure of the proposal. (Where are the
clusters on the sky?)

'''

import matplotlib.pyplot as plt, seaborn as sns
import pandas as pd, numpy as np

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

from math import pi
import pickle, os

from scipy.interpolate import interp1d

global COLORS
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# cite:
# 
# Jaffe, T. J. & Barclay, T. 2017, ticgen: A tool for calculating a TESS
# magnitude, and an expected noise level for stars to be observed by TESS.,
# v1.0.0, Zenodo, doi:10.5281/zenodo.888217
#
# and Stassun & friends (2017).
#import ticgen as ticgen


# # These two, from the website
# # http://dc.zah.uni-heidelberg.de/mwsc/q/clu/form
# # are actually outdated or something. They provided too few resuls..
# close_certain = pd.read_csv('../data/MWSC_search_lt_2000_pc_type_certain.csv')
# close_junk = pd.read_csv('../data/MWSC_search_lt_2000_pc_type_certain.csv')


def get_cluster_data():

    # Downloaded the MWSC from
    # http://cdsarc.u-strasbg.fr/viz-bin/Cat?cat=J%2FA%2BA%2F558%2FA53&target=http&
    tab = Table.read('../data/Kharchenko_2013_MWSC.vot', format='votable')

    df = tab.to_pandas()

    for colname in ['Type', 'Name', 'n_Type', 'SType']:
        df[colname] = [e.decode('utf-8') for e in list(df[colname])]

    # From erratum:
    # For the Sun-like star, a 4 Re planet produces a transit depth of 0.13%. The
    # limiting magnitude for transits to be detectable is about I_C = 11.4 . This
    # also corresponds to K_s ~= 10.6 and a maximum distance of 290 pc, assuming no
    # extinction.

    cinds = np.array(df['d']<500)
    close = df[cinds]
    finds = np.array(df['d']<1000)
    far = df[finds]

    N_c_r0 = int(np.sum(close['N1sr0']))
    N_c_r1 = int(np.sum(close['N1sr1']))
    N_c_r2 = int(np.sum(close['N1sr2']))
    N_f_r0 = int(np.sum(far['N1sr0']))
    N_f_r1 = int(np.sum(far['N1sr1']))
    N_f_r2 = int(np.sum(far['N1sr2']))

    type_d = {'a':'association', 'g':'globular cluster', 'm':'moving group',
              'n':'nebulosity/presence of nebulosity', 'r':'remnant cluster',
              's':'asterism', '': 'no label'}

    ntype_d = {'o':'object','c':'candidate','':'no label'}

    print('*'*50)
    print('\nMilky Way Star Clusters (close := <500pc)'
          '\nN_clusters: {:d}'.format(len(close))+\
          '\nN_stars (in core): {:d}'.format(N_c_r0)+\
          '\nN_stars (in central part): {:d}'.format(N_c_r1)+\
          '\nN_stars (in cluster): {:d}'.format(N_c_r2))


    print('\n'+'*'*50)
    print('\nMilky Way Star Clusters (far := <1000pc)'
          '\nN_clusters: {:d}'.format(len(far))+\
          '\nN_stars (in core): {:d}'.format(N_f_r0)+\
          '\nN_stars (in central part): {:d}'.format(N_f_r1)+\
          '\nN_stars (in cluster): {:d}'.format(N_f_r2))

    print('\n'+'*'*50)

    ####################
    # Post-processing. #
    ####################
    # Compute mean density
    mean_N_star_per_sqdeg = df['N1sr2'] / (pi * df['r2']**2)
    df['mean_N_star_per_sqdeg'] = mean_N_star_per_sqdeg

    # # Compute King profiles
    # king_profiles, theta_profiles = [], []
    # for rt, rc, k, d in zip(np.array(df['rt']),
    #                         np.array(df['rc']),
    #                         np.array(df['k']),
    #                         np.array(df['d'])):

    #     sigma, theta = get_king_proj_density_profile(rt, rc, k, d)
    #     king_profiles.append(sigma)
    #     theta_profiles.append(theta)

    # df['king_profile'] = king_profiles
    # df['theta'] = theta_profiles

    ra = np.array(df['RAJ2000'])
    dec = np.array(df['DEJ2000'])

    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    galactic_long = np.array(c.galactic.l)
    galactic_lat = np.array(c.galactic.b)
    ecliptic_long = np.array(c.barycentrictrueecliptic.lon)
    ecliptic_lat = np.array(c.barycentrictrueecliptic.lat)

    df['galactic_long'] = galactic_long
    df['galactic_lat'] = galactic_lat
    df['ecliptic_long'] = ecliptic_long
    df['ecliptic_lat'] = ecliptic_lat

    cinds = np.array(df['d']<500)
    close = df[cinds]
    finds = np.array(df['d']<1000)
    far = df[finds]

    return close, far, df


def make_wget_script(df):
    '''
    to download stellar data for each cluster, need to run a script of wgets.
    this function makes the script.
    '''

    # get MWSC ids in "0012", "0007" format
    mwsc = np.array(df['MWSC'])
    mwsc_ids = np.array([str(int(f)).zfill(4) for f in mwsc])

    names = np.array(df['Name'])

    f = open('../data/MWSC_stellar_data/get_stellar_data.sh', 'w')

    outstrs = []
    for mwsc_id, name in zip(mwsc_ids, names):
        startstr = 'wget '+\
                 'ftp://cdsarc.u-strasbg.fr/pub/cats/J/A%2BA/558/A53/stars/2m_'
        middlestr = str(mwsc_id) + '_' + str(name)
        endstr = '.dat.bz2 ;\n'
        outstr = startstr + middlestr + endstr
        outstrs.append(outstr)

    f.writelines(outstrs)
    f.close()

    print('made wget script!')



def plot_cluster_positions_scicase(df):
    '''
    Show the positions of d<2kpc clusters, and highlight those with rotation
    period measurements & transiting planets.
    '''

    rotn_clusters = ['NGC_1976', # AKA the orion nebula cluster
                     'NGC_6530',
                     'NGC_2264',
                     'Cep_OB3',
                     'NGC_2362',
                     'NGC_869', # h Per, one of the double cluster
                     'NGC_2547',
                     'IC_2391',
                     'Melotte_20', # alpha Persei cluster, alpha Per
                     'Melotte_22', # AKA Pleiades
                     'NGC_2323', # M 50
                     'NGC_2168', #M 35
                     'NGC_2516',
                     'NGC_1039', #M 34
                     'NGC_2099', # M 37
                     #'NGC_2632', #Praesepe, comment out to avoid overlap
                     #'NGC_6811', #comment out to avoid overlap
                     'NGC_2682' ] #M 67

    transiting_planet_clusters = [
                     'NGC_6811',
                     'NGC_2632' #Praesepe
                    ]

    df = df[df['d'] < 2000]

    df_rotn = df.loc[df['Name'].isin(rotn_clusters)]
    df_rotn = df_rotn[
            ['ecliptic_lat','ecliptic_long','galactic_lat','galactic_long',
            'Name']
            ]

    df_tra = df.loc[df['Name'].isin(transiting_planet_clusters)]

    # Above rotation lists were from Table 1 of Gallet & Bouvier 2015,
    # including M67 which was observed by K2.  Transiting planets from the few
    # papers that have them.  They are cross-matching MWSC's naming scheme. I
    # could not find the Hyades or ScoCen OB.  They both have transiting
    # planets, and the former has rotation studies done.

    c_Hyades = SkyCoord(ra='4h27m', dec=15*u.degree + 52*u.arcminute)
    df_hyades = pd.DataFrame({
            'Name':'Hyades',
            'ecliptic_long':float(c_Hyades.barycentrictrueecliptic.lon.value),
            'ecliptic_lat':float(c_Hyades.barycentrictrueecliptic.lat.value),
            'galactic_long':float(c_Hyades.galactic.l.value),
            'galactic_lat':float(c_Hyades.galactic.b.value)}, index=[0])

    c_ScoOB2 = SkyCoord(ra='16h10m14.73s', dec='-19d19m09.38s') # Mann+2016's position
    df_ScoOB2 = pd.DataFrame({
            'Name':'Sco_OB2',
            'ecliptic_long':float(c_ScoOB2.barycentrictrueecliptic.lon.value),
            'ecliptic_lat':float(c_ScoOB2.barycentrictrueecliptic.lat.value),
            'galactic_long':float(c_ScoOB2.galactic.l.value),
            'galactic_lat':float(c_ScoOB2.galactic.b.value)}, index=[0])

    df_tra = df_tra.append(df_hyades, ignore_index=True)
    df_tra = df_tra.append(df_ScoOB2, ignore_index=True)
    #df_rotn = df_rotn.append(df_hyades, ignore_index=True) #avoid overlap

    # End of data wrangling.


    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap

    for coord in ['galactic','ecliptic']:

        plt.close('all')
        #f, ax = plt.subplots(figsize=(4,4))
        f = plt.figure(figsize=(0.7*5,0.7*4))
        ax = plt.gca()
        m = Basemap(projection='kav7',lon_0=0, resolution='c', ax=ax)

        lats = np.array(df[coord+'_lat'])
        lons = np.array(df[coord+'_long'])
        x, y = m(lons, lats)
        m.scatter(x,y,2,marker='o',facecolor=COLORS[0], zorder=4,
                alpha=0.9,edgecolors=COLORS[0], lw=0)

        lats = np.array(df_rotn[coord+'_lat'])
        lons = np.array(df_rotn[coord+'_long'])
        x, y = m(lons, lats)
        m.scatter(x,y,42,marker='*',color=COLORS[1],edgecolors='k',
                label='have rotation studies', zorder=5,lw=0.4)

        lats = np.array(df_tra[coord+'_lat'])
        lons = np.array(df_tra[coord+'_long'])
        x, y = m(lons, lats)
        m.scatter(x,y,13,marker='s',color=COLORS[1],edgecolors='k',
                label='also have transiting planets', zorder=6, lw=0.45)

        parallels = np.arange(-90.,120.,30.)
        meridians = np.arange(0.,420.,60.)
	# labels = [left,right,top,bottom]
        ps = m.drawparallels(parallels, labels=[1,0,0,0], zorder=2,
                fontsize='x-small')
        ms = m.drawmeridians(meridians, labels=[0,0,0,1], zorder=2,
                fontsize='x-small')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
            box.width, box.height * 0.9])

	# Put a legend below current axis
        #ax.legend(loc='upper center', bbox_to_anchor=(0.01, 0.02),
        #        fancybox=True, ncol=1, fontsize='x-small')

        for _m in ms:
            try:
                #ms[_m][1][0].set_rotation(45)
                if '60' in ms[_m][1][0].get_text():
                    ms[_m][1][0].set_text('')
            except:
                pass
        for _p in ps:
            try:
                if '30' in ps[_p][1][0].get_text():
                    ps[_p][1][0].set_text('')
            except:
                pass

        ax.set_xlabel(coord+' long', labelpad=13, fontsize='x-small')
        ax.set_ylabel(coord+' lat', labelpad=13, fontsize='x-small')

        ######################
	# add TESS footprint #
        ######################
        dat = np.genfromtxt('../data/shemi_nhemi.csv', delimiter=',')
        dat = pd.DataFrame(np.transpose(dat), columns=['icSys', 'tSys', 'teff',
            'logg', 'r', 'm', 'eLat', 'eLon', 'micSys', 'mvSys', 'mic', 'mv',
            'stat', 'nPntg'])
        eLon, eLat = np.array(dat.eLon), np.array(dat.eLat)
        nPntg = np.array(dat.nPntg)
        if coord=='galactic':
            c = SkyCoord(lat=eLat*u.degree, lon=eLon*u.degree,
                    frame='barycentrictrueecliptic')
            lon = np.array(c.galactic.l)
            lat = np.array(c.galactic.b)

        elif coord=='ecliptic':
            lon, lat = eLon, eLat

        nPntg[nPntg >= 4] = 4

        ncolor = 4
        cmap1 = mpl.colors.ListedColormap(
                sns.color_palette("Greys", n_colors=ncolor, desat=1))
        bounds= list(np.arange(0.5,ncolor+1,1))
        norm1 = mpl.colors.BoundaryNorm(bounds, cmap1.N)

        x, y = m(lon, lat)
        out = m.scatter(x,y,s=0.2,marker='s',c=nPntg, zorder=1, cmap=cmap1,
                norm=norm1, rasterized=True, alpha=0.5)
        out = m.scatter(x,y,s=0, marker='s',c=nPntg, zorder=-1, cmap=cmap1,
                norm=norm1, rasterized=True, alpha=1)
        m.drawmapboundary()
        #cbar = f.colorbar(out, cmap=cmap1, norm=norm1, boundaries=bounds,
        #    fraction=0.025, pad=0.05, ticks=np.arange(ncolor)+1,
        #    orientation='vertical')

        #ylabels = np.arange(1,ncolor+1,1)
        #cbarlabels = list(map(str, ylabels))[:-1]
        #cbarlabels.append('$\geq\! 4$')
        #cbar.ax.set_yticklabels(cbarlabels, fontsize='x-small')
        #cbar.set_label('N pointings', rotation=270, labelpad=5, fontsize='x-small')
        #################### 
        f.tight_layout()

        f.savefig('cluster_positions_'+coord+'_scicase.pdf', bbox_inches='tight')



if __name__ == '__main__':

    #make_wget_script(df)

    close, far, df = get_cluster_data()

    plot_cluster_positions_scicase(df)
