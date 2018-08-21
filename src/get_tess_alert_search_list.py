'''
"given a TICID with a transit like event, is the host star interesting in a
non-obvious way?".

independent of SECTOR NUMBER, or WHETHER A STAR FALLS ON TESS SILICON...
this program collects lists of objects that are non-obviously interesting,
and gets TICIDs for them.
'''

import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii

from astroquery.vizier import Vizier

import sys, os, time, re, json
from glob import glob

from mast_utils import tic_single_object_crossmatch, tic_gaia_id_xmatch

def arr(x):
    return np.array(x)

def make_Gagne18_BANYAN_XI_TIC_crossmatch(maxsep=2):
    '''
    Gagne et al 2018's BANYAN XI association list.
    Most of these have inferable ages.

    maxsep = 10 arcsec, preferred (because they're high PM).
    Then use photometry or proper motion information to disambiguate.
    '''
    # Downloaded direct from
    # http://iopscience.iop.org/0004-637X/856/1/23/suppdata/apjaaae09t5_mrt.txt
    t = Table.read('../data/Gagne_2018_apjaaae09t5_mrt.txt',
                   format='ascii.cds')

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
                'Gagne_2018_BANYAN_XI_TIC_crossmatched_{:d}arcsec_maxsep.csv'.format(
                    int(maxsep_arcsec)),
                format='ecsv')


def plot_xmatch_separations(t, outname):
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.hist(t[t['dstArcSec'] != -99]['dstArcSec'], bins=40)
    plt.xlabel('xmatch separation (arcsec)')
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    print('saved {:s}'.format(outname))


def make_Gagne18_skymaps(t):

    from tessmaps import plot_skymaps_of_targets as pst
    gd = t[t['dstArcSec'] != -99]
    bad = t[t['dstArcSec'] == -99]

    pst.plot_mwd(gd['ra'], gd['dec'], 'black',
                 title='Gagne+18 successful xmatch',
                 savname='Gagne18_succeeded_xmatch_pts.pdf',
                 is_radec=True)

    pst.plot_mwd(bad['ra'], bad['dec'], 'black',
                 title='Gagne+18 failed xmatch',
                 savname='Gagne18_failed_xmatch_pts.pdf',
                 is_radec=True)


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



def crossmatch_alerts(ticidlist_path, sector_id=0, find_alerts_in_MWSC=True):
    '''
    take a list of TIC IDs from MIT alerts. do they overlap with any of:
        * Kharchenko+13's clusters
        * Gagne+18's associations
        * Kane's known planet list

    args: ticidlist_path, a path to a newline-separated list of TICIDs of
        interest. no header info is assumed.
    '''

    df = pd.read_csv(ticidlist_path, names=['ticid'], comment='#')
    TOIids = arr(df['ticid'])

    # make list of all files with TIC crossmatches that we will search
    searchfiles = []
    searchfiles.append('../results/Gagne_2018_BANYAN_XI_TIC_crossmatched_10arcsec_maxsep.csv')
    searchfiles.append('../data/Kane_MAST_Crossmatch_CTL.csv')
    searchfiles.append('../results/Oh_2017_TIC_crossmatched_2arcsec_on_MAST.csv')

    searchfiles.append('../results/Rizzuto_15_table_2_USco_PMS_TIC_crossmatched_3arcsec_MAST.csv')
    searchfiles.append('../results/Rizzuto_15_table_3_USco_disks_TIC_crossmatched_3arcsec_MAST.csv')
    searchfiles.append('../results/Rizzuto_11_table_1_ScoOB2_members_TIC_3arcsec_crossmatch_MAST.csv')
    searchfiles.append('../results/Preibisch_01_table_1_USco_LiRich_members_TIC_3arcsec_crossmatch_MAST.csv')
    searchfiles.append('../results/Luhman_12_table_1_USco_IR_excess_TIC_3arcsec_crossmatch_MAST.csv')

    if find_alerts_in_MWSC:
        mtxdir = '../results/MWSC_TIC_crossmatched/'
        for f in glob(mtxdir+'????_*_1sigma_members_TIC_crossmatched.csv'):
            searchfiles.append(f)

    searchfiles = np.sort(searchfiles)

    savname = 'sector_{:d}_alert_matches.csv'.format(sector_id)
    savdir = '../results/alert_search_matches/'
    if os.path.exists(savdir+savname):
        os.remove(savdir+savname)

    # crossmatch on TICIDs
    for sf in searchfiles:

        # extended csv format assumed
        t = ascii.read(sf)

        t_inds = np.in1d(t['MatchID'], TOIids)

        if len(t[t_inds])>0:

            overlap = t[t_inds]

            print('got {:d} matches in {:s}'.format(len(overlap), sf))

            df = pd.DataFrame(
                {'ticid':overlap['MatchID'],
                 'dstArcSec':overlap['dstArcSec'],
                 'path':np.repeat(sf,len(overlap))
                })

            if not os.path.exists(savdir+savname):
                df.to_csv(savdir+savname,index=False,header='column_names')
            else:
                df.to_csv(savdir+savname,index=False,header=False,mode='a')

    print('done, saved in {:s}'.format(savdir+savname))


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

if __name__ == '__main__':

    do_Kharchenko13 = False
    do_Gagne18 = False
    do_Oh17 = False
    do_Rizzuto15 = False
    do_Rizzuto11 = False
    do_Preibisch01 = False
    do_Luhman12 = True

    make_plots = False

    find_which_alerts_are_interesting = True #TODO: run w/ real data
    find_alerts_in_MWSC = False # added option b/c MWSC parsing is slow

    g18_maxsep = 10 # arcsec

    if do_Gagne18:
        make_Gagne18_BANYAN_XI_TIC_crossmatch(maxsep=g18_maxsep)

        if make_plots:
            t = ascii.read(
                '../results/Gagne_2018_BANYAN_XI_TIC_crossmatched_{:d}arcsec_maxsep.csv'.
                format(g18_maxsep))
            plot_xmatch_separations(t,
                        '../results/Gagne18_BANYAN_XI_TIC_crossmatch_separations.png')
            make_Gagne18_skymaps(t)

    if do_Kharchenko13:
        make_Kharchenko13_TIC_crossmatch()

        if make_plots:
            mwsc_xmatch_dir = '../results/MWSC_TIC_crossmatched/'
            fnames = glob(mwsc_xmatch_dir+'01?[0-1]_*_1sigma_members_TIC_crossmatched.csv')
            for fname in fnames:
                t = ascii.read(fname)
                plot_xmatch_separations(
                    t, fname.replace('.csv','_separationhist.png'))

    if do_Oh17:
        make_Oh17_TIC_crossmatch()
    if do_Rizzuto15:
        make_Rizzuto15_TIC_crossmatch()
    if do_Rizzuto11:
        make_Rizzuto11_TIC_crossmatch()
    if do_Preibisch01:
        make_Preibisch01_TIC_crossmatch()
    if do_Luhman12:
        make_Luhman12_TIC_crossmatch()

    if find_which_alerts_are_interesting:

        if not find_alerts_in_MWSC:
            print(50*'#')
            print('WARNING: ignoring MWSC lists!')
            print(50*'#')

        crossmatch_alerts('../data/sector_0_TOI_list.txt',
                          find_alerts_in_MWSC=find_alerts_in_MWSC)
