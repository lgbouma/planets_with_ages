# -*- coding: utf-8 -*-
'''
functions to download and wrangle catalogs of members that I want to crossmatch
against TESS objects of interest (alerts)

mostly called from `get_tess_alert_search_list.py`
'''
from __future__ import division, print_function

import numpy as np, pandas as pd

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii

from astroquery.vizier import Vizier

import sys, os, re
from glob import glob

from mast_utils import tic_single_object_crossmatch

from numpy import array as arr

def crossmatch_MWSC_to_TIC(maxsep=(2*u.arcsec).to(u.deg).value):

    mwsc_1sig_files = np.sort(glob(
        '../data/MWSC_1sigma_members/????_*_1sigma_members.csv'))

    for mwsc_file in mwsc_1sig_files:

        outpath = '../results/MWSC_TIC_crossmatched/{:s}_TIC_crossmatched.csv'.format(
            mwsc_file.split('/')[-1].replace('.csv',''))
        if os.path.exists(outpath):
            print('found {:s}, continue'.format(outpath))
            continue

        t = ascii.read(mwsc_file)
        ra, dec = t['RAdeg'], t['DEdeg']

        cols = ['dstArcSec', 'Tmag', 'GAIA', 'Teff', 'MatchID', 'MatchRA',
                'MatchDEC', 'ra', 'dec']

        sav = {}
        for col in cols:
            sav[col] = []

        print('{:d} objects in {:s}'.format(len(ra), mwsc_file))

        for ix, _ra, _dec in list(zip(range(len(ra)), ra, dec)):

            print('{:d}/{:d}'.format(ix, len(ra)))
            xm = tic_single_object_crossmatch(_ra, _dec, maxsep)

            if len(xm['data'])==0:
                for k in list(sav.keys()):
                    if k=='ra':
                        sav['ra'].append(_ra)
                    elif k=='dec':
                        sav['dec'].append(_dec)
                    else:
                        sav[k].append(-99)
                continue

            elif len(xm['data'])==1:
                for k in list(sav.keys()):
                    sav[k].append(xm['data'][0][k])
                continue

            else:
                # take the closest star as the match.
                sep_distances = []
                for dat in xm['data']:
                    sep_distances.append(dat['dstArcSec'])
                sep_distances = np.array(sep_distances)
                closest_ind = np.argsort(sep_distances)[0]
                print(sep_distances, sep_distances[closest_ind])
                for k in list(sav.keys()):
                    sav[k].append(xm['data'][closest_ind][k])

            del xm

        for k in list(sav.keys()):
            t[k] = np.array(sav[k])


        ascii.write(t, output=outpath, format='ecsv')
        print('saved {:s}'.format(outpath))

    print('done')


def make_Kharchenko13_TIC_crossmatch():
    '''
    Kharchenko's list of stars in clusters.

    maxsep = 2 arcsec, hard-coded by default.
    '''

    from parse_MWSC import get_cluster_data, get_MWSC_stellar_data, \
        make_wget_script

    close, far, df = get_cluster_data()

    sdatadir = '/home/luke/local/MWSC_stellar_data/'

    if not os.path.exists(sdatadir+'get_stellar_data.sh'):
        make_wget_script(df)

    if not os.path.exists(sdatadir+'2m_0001_Berkeley_58.dat'):
        raise AssertionError(
            'u must manually execute data collection script and extract')

    get_MWSC_stellar_data(df, '1sigma_members', p_0=61,
                          datadir='/home/luke/local/MWSC_stellar_data/')

    crossmatch_MWSC_to_TIC()


def make_Oh17_TIC_crossmatch():
    '''
    Semyeong Oh et al (2017) discovered 10.6k stars within 10pc that are in
    likely comoving pairs.

    see
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/153/257
    '''

    # Download Oh's tables of stars, pairs, and groups.
    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/AJ/153/257')
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    stars = catalogs[0]
    pairs = catalogs[1]
    groups = catalogs[2]

    stars['RA'] = stars['RAJ2000']
    stars['DEC'] = stars['DEJ2000']
    foo = stars.to_pandas()
    foo.to_csv('../data/Oh_2017_table1a_stars.csv',index=False)

    print('saved ../data/Oh_2017_table1a_stars.csv')
    print(
    '''I then uploaded this list to MAST, and used their spatial
        cross-matching with a 2 arcsecond cap, following
            https://archive.stsci.edu/tess/tutorials/upload_list.html
        This crossmatch the output that I then saved to
            ../results/Oh_2017_TIC_crossmatched_2arcsec_on_MAST.csv
        and it was a good deal faster than the object-by-object crossmatch that
        I ran through their API for the other two searches.

        Of the 10606 stars from Oh 2017, there were 9701 matches. Good enough.
    '''
    )



def make_Rizzuto15_TIC_crossmatch():
    '''
    Aaron Rizzuto et al (2015) picked out ~400 candidate USco members, then
    surveyed them for Li absorption. Led to 237 spectroscopically confirmed
    members.

    see
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/MNRAS/448/2737
    '''

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/MNRAS/448/2737')
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    cands = catalogs[0]
    usco_pms = catalogs[1] # pre-MS stars in USco
    usco_disk = catalogs[2] # members of USco w/ circumstellar disk

    c = SkyCoord([ra.replace(' ',':')
                    for ra in list(map(str,usco_pms['RAJ2000']))],
                 [de.replace(' ',':')
                    for de in list(map(str,usco_pms['DEJ2000']))],
                 unit=(u.hourangle, u.deg))
    usco_pms['RA'] = c.ra.value
    usco_pms['DEC'] = c.dec.value
    usco_pms.remove_column('RAJ2000')
    usco_pms.remove_column('DEJ2000')
    foo = usco_pms.to_pandas()
    outname = '../data/Rizzuto_15_table_2_USco_PMS.csv'
    import IPython; IPython.embed()

    foo.to_csv(outname,index=False)
    print('saved {:s}'.format(outname))

    c = SkyCoord([ra.replace(' ',':')
                    for ra in list(map(str,usco_disk['RAJ2000']))],
                 [de.replace(' ',':')
                    for de in list(map(str,usco_disk['DEJ2000']))],
                 unit=(u.hourangle, u.deg))
    usco_disk['RA'] = c.ra.value
    usco_disk['DEC'] = c.dec.value
    usco_disk.remove_column('RAJ2000')
    usco_disk.remove_column('DEJ2000')
    foo = usco_disk.to_pandas()
    outname = '../data/Rizzuto_15_table_3_USco_hosts_disk.csv'
    foo.to_csv(outname,index=False)
    print('saved {:s}'.format(outname))

    print(
    ''' I then uploaded these lists to MAST, and used their spatial
        cross-matching with a 3 arcsecond cap, following
            https://archive.stsci.edu/tess/tutorials/upload_list.html

        This crossmatch is the output that I then saved to
            '../results/Rizzuto_15_table_2_USco_PMS_TIC_crossmatched_2arcsec_MAST.csv'
            '../results/Rizzuto_15_table_3_USco_disks_TIC_crossmatched_2arcsec_MAST.csv'
    '''
    )


def make_Luhman12_TIC_crossmatch():
    '''
    Luhman and Mamajek 2012 combined Spitzer & WISE photometry for all known
    USco members. Found IR excesses.
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/ApJ/758/31
    '''
    vizier_search_str = 'J/ApJ/758/31'
    table_num = 0
    ra_str = '_RA'
    dec_str = '_DE'
    outname = '../data/Luhman_12_table_1_USco_IR_excess.csv'

    make_vizier_TIC_crossmatch(vizier_search_str, ra_str, dec_str,
                               table_num=table_num, outname=outname)


def make_Preibisch01_TIC_crossmatch():
    '''
    Thomas Preibisch did a spectroscopic survey for low-mass members of USco.
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/121/1040
    '''
    vizier_search_str = 'J/AJ/121/1040'
    table_num = 0
    ra_str = '_RA'
    dec_str = '_DE'
    outname = '../data/Preibisch_01_table_1_USco_LiRich_members.csv'

    make_vizier_TIC_crossmatch(vizier_search_str, ra_str, dec_str,
                               table_num=table_num, outname=outname)


def make_Rizzuto11_TIC_crossmatch():
    '''
    Aaron Rizzuto et al (2011) gave a list of 436 Sco OB2 members.
    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/MNRAS/416/3108
    '''
    vizier_search_str = 'J/MNRAS/416/3108'
    table_num = 0
    ra_str = '_RA'
    dec_str = '_DE'
    outname = '../data/Rizzuto_11_table_1_ScoOB2_members.csv'

    make_vizier_TIC_crossmatch(vizier_search_str, ra_str, dec_str,
                               table_num=table_num, outname=outname)


def make_Kraus14_TIC_crossmatch():
    '''
    Adam Kraus et al (2014) did spectroscopy of members of the
    Tucana-Horologium moving group, looking at RVs, Halpha emission, and Li
    absoprtion.

    WARNING: only ~70% of the rows in this table turned out to be members.

    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/AJ/147/146
    '''
    vizier_search_str = 'J/AJ/147/146'
    table_num = 1
    ra_str = '_RA'
    dec_str = '_DE'
    outname = '../data/Kraus_14_table_2_TucanaHorologiumMG_members.csv'

    make_vizier_TIC_crossmatch(vizier_search_str, ra_str, dec_str,
                               table_num=table_num, outname=outname)


def make_Roser11_TIC_crossmatch():
    '''
    Roser et al (2011) used PPMXL (positions, propoer motions, and photometry
    for 9e8 stars) to report Hyades members down to r<17.

    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+A/531/A92
    '''
    vizier_search_str = 'J/A+A/531/A92'
    table_num = 0
    ra_str = 'RAJ2000'
    dec_str = 'DEJ2000'
    outname = '../data/Roser11_table_1_Hyades_members.csv'

    make_vizier_TIC_crossmatch(vizier_search_str, ra_str, dec_str,
                               table_num=table_num, outname=outname)


def make_Casagrande_11_TIC_crossmatch():
    '''
    Casagrande et al (2011) re-analyzed the Geneva-Copenhagen survey, and got a
    kinematically unbiased sample of solar neighborhood stars with kinematics,
    metallicites, and ages.

    If Casagrande's reported max likelihood ages (from Padova isochrones) are
    below 1 Gyr, then that's interesting enough to look into.

    NOTE that this age cut introduces serious biases into the stellar mass
    distribution -- see Casagrande+2011, Figure 14.

    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+A/530/A138
    '''
    vizier_search_str = 'J/A+A/530/A138'
    table_num = 0
    ra_str = 'RAJ2000'
    dec_str = 'DEJ2000'
    outname = '../data/Casagrande_2011_table_1_GCS_ages_lt_1Gyr.csv'

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs(vizier_search_str)
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    tab = catalogs[0]

    # ageMLP: max-likelihood padova isochrone ages
    # ageMLB: max-likelihood BASTI isochrone ages. not queried.
    sel = (tab['ageMLP'] > 0)
    sel &= (tab['ageMLP'] < 1)

    coords = SkyCoord(ra=tab[ra_str], dec=tab[dec_str], frame='icrs',
                      unit=(u.hourangle, u.deg))
    # MAST uploads need these two column names
    tab['RA'] = coords.ra.value
    tab['DEC'] = coords.dec.value
    tab.remove_column('RAJ2000')
    tab.remove_column('DEJ2000')

    foo = tab[sel].to_pandas()
    foo.to_csv(outname,index=False)
    print('saved {:s}'.format(outname))

    print(
    ''' I then uploaded these lists to MAST, and used their spatial
        cross-matching with a 3 arcsecond cap, following
            https://archive.stsci.edu/tess/tutorials/upload_list.html

        This crossmatch is the output that I then saved to
            {:s}
    '''.format(outname.replace('data','results').replace('.csv','_TIC_3arcsec_crossmatch_MAST.csv'))
    )


def make_vizier_TIC_crossmatch(vizier_search_str, ra_str, dec_str, table_num=0,
                              outname=''):
    '''
    vizier_search_str: 'J/MNRAS/416/3108'
    table_num=0
    ra_str = '_RA'
    dec_str = '_DE'
    outname = '../data/Rizzuto_11_table_1_ScoOB2_members.csv'
    '''

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs(vizier_search_str)
    catalogs = Vizier.get_catalogs(catalog_list.keys())

    tab = catalogs[0]

    # MAST uploads need these two column names
    tab['RA'] = tab[ra_str]
    tab['DEC'] = tab[dec_str]

    assert tab[ra_str].unit == u.deg

    foo = tab.to_pandas()
    foo.to_csv(outname,index=False)
    print('saved {:s}'.format(outname))

    print(
    ''' I then uploaded these lists to MAST, and used their spatial
        cross-matching with a 3 arcsecond cap, following
            https://archive.stsci.edu/tess/tutorials/upload_list.html

        This crossmatch is the output that I then saved to
            {:s}
    '''.format(outname.replace('data','results').replace('.csv','_TIC_3arcsec_crossmatch_MAST.csv'))
    )


def make_Gagne18_BANYAN_XIII_TIC_crossmatch():

    # Downloaded direct from
    # http://iopscience.iop.org/0004-637X/862/2/138/suppdata/apjaaca2et2_mrt.txt
    # God I wish vizier were a thing.
    tablepath = '../data/Gagne_2018_BANYAN_XIII_apjaaca2et2_mrt.txt'

    make_Gagne18_BANYAN_any_TIC_crossmatch(
        tablepath, namestr='Gagne_2018_BANYAN_XIII_TIC_crossmatched',
        maxsep=10)


def make_Gagne18_BANYAN_XII_TIC_crossmatch():

    # Downloaded direct from
    # http://iopscience.iop.org/0004-637X/860/1/43/suppdata/apjaac2b8t4_mrt.txt
    tablepath = '../data/Gagne_2018_BANYAN_XII_apjaac2b8t4_mrt.txt'

    make_Gagne18_BANYAN_any_TIC_crossmatch(
        tablepath, namestr='Gagne_2018_BANYAN_XII_TIC_crossmatched',
        maxsep=10)


def make_Gagne18_BANYAN_XI_TIC_crossmatch():

    # Downloaded direct from
    # http://iopscience.iop.org/0004-637X/856/1/23/suppdata/apjaaae09t5_mrt.txt
    tablepath = '../data/Gagne_2018_apjaaae09t5_mrt.txt'

    make_Gagne18_BANYAN_any_TIC_crossmatch(
        tablepath, namestr='Gagne_2018_BANYAN_XI_TIC_crossmatched',
        maxsep=10)


def make_Gagne18_BANYAN_any_TIC_crossmatch(
        tablepath,
        namestr=None,
        maxsep=10):
    '''
    J. Gagne's tables have a particular format that requires some wrangling.
    Also, since so many of the stars are high PM, the spatial cross-matches
    will be crap unless we also include PM information in the matching.

    This means it's best to use the MAST API, rather than the MAST Portal.
    All that code, not for nothing!
    '''
    assert type(namestr) == str
    t = Table.read(tablepath, format='ascii.cds')

    RAh, RAm, RAs = arr(t['RAh']), arr(t['RAm']), arr(t['RAs'])

    RA_hms =  [str(rah).zfill(2)+'h'+
               str(ram).zfill(2)+'m'+
               str(ras).zfill(2)+'s'
               for rah,ram,ras in zip(RAh, RAm, RAs)]

    DEd, DEm, DEs = arr(t['DEd']),arr(t['DEm']),arr(t['DEs'])
    DEsign = arr(t['DE-'])
    DEsign[DEsign != '-'] = '+'

    DE_dms = [str(desgn)+
              str(ded).zfill(2)+'d'+
              str(dem).zfill(2)+'m'+
              str(des).zfill(2)+'s'
              for desgn,ded,dem,des in zip(DEsign, DEd, DEm, DEs)]

    coords = SkyCoord(ra=RA_hms, dec=DE_dms, frame='icrs')

    RA = coords.ra.value
    dec = coords.dec.value
    pm_RA, pm_dec = arr(t['pmRA']), arr(t['pmDE'])

    # get TIC IDs for these stars
    maxsep_arcsec = maxsep
    maxsep = (maxsep*u.arcsec).to(u.deg).value

    # MatchID means TIC ID. cols and 
    cols = ['dstArcSec', 'Tmag', 'GAIA', 'Teff', 'MatchID', 'MatchRA',
            'MatchDEC', 'ra', 'dec']

    sav = {}
    for col in cols:
        sav[col] = []

    print('{:d} objects in Gagne table'.format(len(RA)))

    for ix, _ra, _dec, _pmra, _pmdec in list(
        zip(range(len(RA)), RA, dec, pm_RA, pm_dec)):

        print('{:d}/{:d}'.format(ix, len(RA)))
        xm = tic_single_object_crossmatch(_ra, _dec, maxsep)

        if len(xm['data'])==0:
            for k in list(sav.keys()):
                if k=='ra':
                    sav['ra'].append(_ra)
                elif k=='dec':
                    sav['dec'].append(_dec)
                else:
                    sav[k].append(-99)
            continue

        elif len(xm['data'])==1:
            for k in list(sav.keys()):
                sav[k].append(xm['data'][0][k])
            continue

        else:
            # usually, would take the closest star as the match.
            sep_distances, pms_ra, pms_dec = [], [], []
            for dat in xm['data']:
                sep_distances.append(dat['dstArcSec'])
                pms_ra.append(dat['pmRA'])
                pms_dec.append(dat['pmDEC'])

            sep_distances = np.array(sep_distances).astype(float)
            pms_ra = np.array(pms_ra).astype(float)
            pms_dec = np.array(pms_dec).astype(float)
            closest_sep_ind = np.argsort(sep_distances)[0]

            print('\tdesired propmot_ra, propmot_dec')
            print('\t', _pmra, _pmdec)
            print('\tmatched sep distances, propmot_ra, propmot_dec')
            print('\t', sep_distances, pms_ra, pms_dec)

            # however, w/ 10 arcsec xmatch radius, we really want the star
            # that is closest, unless it has a bad PM match...

            if (
                (type(_pmra) != np.float64 and type(_pmdec) != np.float64)
                or
                (_pmra == 0. and _pmdec == 0.)
            ):
                # if we don't have reference proper motions, we're screwed
                # anyway, so just take the closest star.
                print('\ttaking closest in separation b/c bad reference PM')
                for k in list(sav.keys()):
                    sav[k].append(xm['data'][closest_sep_ind][k])
                continue

            # compute the expected magnitude of proper motion
            mu_expected = np.sqrt( _pmdec**2 + _pmra**2 * np.cos(_dec)**2 )
            # compute the magnitude of proper motions in matches. (the
            # declinations are close enough that the slightly-wrong projection
            # does not matter).
            mu_matches = np.sqrt(pms_dec**2 + pms_ra**2 * np.cos(_dec)**2)

            try:
                closest_mu_ind = np.nanargmin(mu_matches - mu_expected)
            except ValueError:
                print('\ttaking closest in separation b/c matched PMs all nan')
                for k in list(sav.keys()):
                    sav[k].append(xm['data'][closest_sep_ind][k])
                continue


            if len(mu_matches[~np.isfinite(mu_matches)]) > 0:
                print('\tcaught at least one nan in matched PMs')
                #import IPython; IPython.embed()

            if closest_sep_ind == closest_mu_ind:
                print('\ttaking closest in separation (same as in PM)')
                for k in list(sav.keys()):
                    sav[k].append(xm['data'][closest_sep_ind][k])
                continue

            # if the closest in separation and closest in PM differ, but the
            # closest in separation has the wrong PM sign, then take the
            # closest in PM.
            elif ( (closest_sep_ind != closest_mu_ind) and
                    ( (np.sign(pms_ra[closest_sep_ind]) != np.sign(_pmra)) or
                      (np.sign(pms_dec[closest_sep_ind]) != np.sign(_pmdec)) )
                 ):

                print('\ttaking closest in PM magnitude (bad signs for closest in sep)')
                for k in list(sav.keys()):
                    sav[k].append(xm['data'][closest_mu_ind][k])
                continue

            # if the closest in separation and closest in PM differ, and the
            # closest in separation has the right PM signs, then take the
            # closest in separation.
            # (ideally chi_sq would be in play?)
            elif ( (closest_sep_ind != closest_mu_ind) and
                    ( (np.sign(pms_ra[closest_sep_ind]) == np.sign(_pmra)) and
                      (np.sign(pms_dec[closest_sep_ind]) == np.sign(_pmdec)) )
                 ):

                print('\ttaking closest in separation (good signs for closest in sep)')
                for k in list(sav.keys()):
                    sav[k].append(xm['data'][closest_sep_ind][k])
                continue

            # if the closest in separation and closest in PM differ, and the
            # closest in separation has one wrong PM sign, then take the
            # closest in PM.
            # (ideally chi_sq would be in play?)
            elif ( (closest_sep_ind != closest_mu_ind) and
                    ( ((np.sign(pms_ra[closest_sep_ind]) == np.sign(_pmra)) and
                      (np.sign(pms_dec[closest_sep_ind]) != np.sign(_pmdec)))
                     or
                      ((np.sign(pms_ra[closest_sep_ind]) != np.sign(_pmra)) and
                      (np.sign(pms_dec[closest_sep_ind]) == np.sign(_pmdec)))
                    )
                 ):

                print('\ttaking closest in PM magnitude b/c sep signs wrong')
                for k in list(sav.keys()):
                    sav[k].append(xm['data'][closest_mu_ind][k])
                continue

            else:
                raise AssertionError

        del xm


    for k in list(sav.keys()):
        t[k] = np.array(sav[k])

    ascii.write(t,
                output='../results/'+\
                '{:s}_{:d}arcsec_maxsep.csv'.format(namestr,
                    int(maxsep_arcsec)),
                format='ecsv')


def make_Schalufman14_TIC_crossmatch():

    print('nothing needed; the files he sent were ready to cross-mach on MAST')

    print('saved to '
          '../results/Schlaufman14_lowmet_highFP_'
          'rate_TIC_3arcsec_crossmatch_MAST.csv')
    print('saved to '
          '../results/Schlaufman14_lowmet_lowFP_'
          'rate_TIC_3arcsec_crossmatch_MAST.csv')
    pass


def make_Bell17_TIC_crossmatch():

    with open('../data/Bell_2017_32Ori_table_3.txt') as f:
        lines = f.readlines()

    lines = [l.replace('\n','') for l in lines if not l.startswith('#') and
             len(l) > 200]

    twomass_id_strs = []
    for l in lines:
        try:
            twomass_id_strs.append(
                re.search('[0-9]{8}.[0-9]{7}', l).group(0)
            )
        except:
            print('skipping')
            print(l)
            continue

    RA = [t[0:2]+'h'+t[2:4]+'m'+t[4:6]+'.'+t[6:8]
              for t in twomass_id_strs
         ]

    DE = [t[8]+t[9:11]+'d'+t[11:13]+'m'+t[13:15]+'.'+t[15]
              for t in twomass_id_strs
         ]

    c = SkyCoord(RA, DE, frame='icrs')

    # MAST uploads need these two column names
    foo = pd.DataFrame()
    foo['RA'] = c.ra.value
    foo['DEC'] = c.dec.value

    outname = '../data/Bell_2017_32Ori_table_3_positions.csv'
    foo.to_csv(outname,index=False)
    print('saved {:s}'.format(outname))

    print(
    ''' I then uploaded these lists to MAST, and used their spatial
        cross-matching with a 3 arcsecond cap, following
            https://archive.stsci.edu/tess/tutorials/upload_list.html

        This crossmatch is the output that I then saved to
            {:s}
    '''.format(
        outname.
        replace('data','results').
        replace('.csv','_TIC_3arcsec_crossmatch_MAST.csv'))
    )



