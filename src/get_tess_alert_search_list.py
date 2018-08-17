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

import sys, os, time, re, json
from glob import glob

from mast_utils import tic_single_object_crossmatch

def arr(x):
    return np.array(x)

def make_Gagne18_TIC_crossmatch():
    '''
    Gagne et al 2018's BANYAN association list.
    Most of these have inferable ages.

    maxsep = 2 arcsec, hard-coded by default.
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

    # get TIC IDs for these stars
    maxsep = (2*u.arcsec).to(u.deg).value

    # MatchID means TIC ID. cols and 
    cols = ['dstArcSec', 'Tmag', 'GAIA', 'Teff', 'MatchID', 'MatchRA',
            'MatchDEC', 'ra', 'dec']

    sav = {}
    for col in cols:
        sav[col] = []

    print('{:d} objects in Gagne table'.format(len(RA)))

    for ix, thisra, thisdec in list(zip(range(len(RA)), RA, dec)):

        print('{:d}/{:d}'.format(ix, len(RA)))
        xm = tic_single_object_crossmatch(thisra, thisdec, maxsep)

        if len(xm['data'])==0:
            for k in list(sav.keys()):
                if k=='ra':
                    sav['ra'].append(thisra)
                elif k=='dec':
                    sav['dec'].append(thisdec)
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

    ascii.write(t, output='../results/Gagne_2018_TIC_crossmatched.csv',
                format='ecsv')


def plot_xmatch_separations(t, outname):
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.hist(t[t['dstArcSec'] != -99]['dstArcSec'], bins=20)
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

        for ix, thisra, thisdec in list(zip(range(len(ra)), ra, dec)):

            print('{:d}/{:d}'.format(ix, len(ra)))
            xm = tic_single_object_crossmatch(thisra, thisdec, maxsep)

            if len(xm['data'])==0:
                for k in list(sav.keys()):
                    if k=='ra':
                        sav['ra'].append(thisra)
                    elif k=='dec':
                        sav['dec'].append(thisdec)
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


def crossmatch_alerts(ticidlist_path, sector_id=0):
    '''
    take a list of TIC IDs from MIT alerts. do they overlap with any of:
        * Kharchenko+13's clusters
        * Gagne+18's associations
        * Kane's known planet list

    args: ticidlist_path, a path to a newline-separated list of TICIDs of
        interest. no header info is assumed.
    '''

    df = pd.read_csv(ticidlist_path, names=['ticid'])
    TOIids = arr(df['ticid'])

    # make list of all files with TIC crossmatches that we will search
    searchfiles = []
    searchfiles.append('../results/Gagne_2018_TIC_crossmatched.csv')
    searchfiles.append('../data/Kane_MAST_Crossmatch_CTL.csv')
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


if __name__ == '__main__':

    make_plots = False

    if not os.path.exists('../results/Gagne_2018_TIC_crossmatched.csv'):
        make_Gagne18_TIC_crossmatch()

    if make_plots:
        t = ascii.read('../results/Gagne_2018_TIC_crossmatched.csv')
        plot_xmatch_separations(t, '../results/Gagne18_TIC_crossmatch_separations.png')
        make_Gagne18_skymaps(t)

    #make_Kharchenko13_TIC_crossmatch()

    if make_plots:
        mwsc_xmatch_dir = '../results/MWSC_TIC_crossmatched/'
        fnames = glob(mwsc_xmatch_dir+'01?[0-1]_*_1sigma_members_TIC_crossmatched.csv')
        for fname in fnames:
            t = ascii.read(fname)
            plot_xmatch_separations(
                t, fname.replace('.csv','_separationhist.png'))

    #TODO: use real data, once it exists
    crossmatch_alerts('../data/sector_0_TOI_list.txt')
