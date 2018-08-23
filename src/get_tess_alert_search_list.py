# -*- coding: utf-8 -*-
'''
Given a TICID with a transit like event, is the host star interesting in a
non-obvious way?

This program collects lists of objects that I find interesting, and gets TICIDs
for them. It also runs the cross-match against an input list of alert TIC IDs.

Look for TIC ID alert matches in:

    * Stephen Kane's known planet list.
    * Kharchenko+2013 MWSC 1 sigma cluster members.

    An assortment of young-star / moving group lists:
    * Bell_2017_32Ori_table_3_positions_TIC_3arcsec_crossmatch_MAST.csv
    * Gagne_2018_BANYAN_XIII_TIC_crossmatched_10arcsec_maxsep.csv
    * Gagne_2018_BANYAN_XII_TIC_crossmatched_10arcsec_maxsep.csv
    * Gagne_2018_BANYAN_XI_TIC_crossmatched_10arcsec_maxsep.csv
    * Luhman_12_table_1_USco_IR_excess_TIC_3arcsec_crossmatch_MAST.csv
    * Oh_2017_TIC_crossmatched_2arcsec_on_MAST.csv
    * Preibisch_01_table_1_USco_LiRich_members_TIC_3arcsec_crossmatch_MAST.csv
    * Rizzuto_11_table_1_ScoOB2_members_TIC_3arcsec_crossmatch_MAST.csv
    * Rizzuto_15_table_2_USco_PMS_TIC_crossmatched_3arcsec_MAST.csv
    * Rizzuto_15_table_3_USco_disks_TIC_crossmatched_3arcsec_MAST.csv
    * Roser11_table_1_Hyades_members_TIC_3arcsec_crossmatch_MAST.csv

    Other odd star lists:
    * Schlaufman14_lowmet_highFP_rate_TIC_3arcsec_crossmatch_MAST.csv
    * Schlaufman14_lowmet_lowFP_rate_TIC_3arcsec_crossmatch_MAST.csv
'''
from __future__ import division, print_function

import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii

from astroquery.vizier import Vizier

import os
from glob import glob

from mast_utils import tic_single_object_crossmatch

from crossmatch_catalogs_vs_TIC import make_Gagne18_BANYAN_XI_TIC_crossmatch, \
    make_Kharchenko13_TIC_crossmatch, make_Luhman12_TIC_crossmatch, \
    make_Oh17_TIC_crossmatch, make_Preibisch01_TIC_crossmatch, \
    make_Rizzuto11_TIC_crossmatch, make_Gagne18_BANYAN_XII_TIC_crossmatch, \
    make_Gagne18_BANYAN_XIII_TIC_crossmatch, make_Bell17_TIC_crossmatch, \
    make_Kraus14_TIC_crossmatch, make_Roser11_TIC_crossmatch, \
    make_Schalufman14_TIC_crossmatch

from crossmatch_catalogs_vs_TIC import make_vizier_TIC_crossmatch

from numpy import array as arr

#############################
# Plotting helper functions #
#############################
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
    searchfiles = np.sort(glob('../results/*crossmatch*.csv'))

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

        print('searching', sf)

        # extended csv format assumed. if regular csv, that works too.
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

        print('\n')

    print('done, saved in {:s}'.format(savdir+savname))


if __name__ == '__main__':

    do_Kharchenko13 = False
    do_BANYAN_XI = False
    do_Oh17 = False
    do_Rizzuto15 = False
    do_Rizzuto11 = False
    do_Preibisch01 = False
    do_Luhman12 = False
    do_BANYAN_XII= False
    do_BANYAN_XIII= False
    do_Bell_17 = False
    do_Kraus_14 = False
    do_Roser_11 = False
    do_Schlaufman_14 = False

    find_which_alerts_are_interesting = True #TODO: run w/ real data
    find_alerts_in_MWSC = False # added option b/c MWSC parsing is slow

    make_diagnostic_plots = False

    if do_Kharchenko13:
        make_Kharchenko13_TIC_crossmatch()
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
    if do_BANYAN_XI:
        make_Gagne18_BANYAN_XI_TIC_crossmatch()
    if do_BANYAN_XII:
        make_Gagne18_BANYAN_XII_TIC_crossmatch()
    if do_BANYAN_XIII:
        make_Gagne18_BANYAN_XIII_TIC_crossmatch()
    if do_Bell_17:
        make_Bell17_TIC_crossmatch()
    if do_Kraus_14:
        make_Kraus14_TIC_crossmatch()
    if do_Roser_11:
        make_Roser11_TIC_crossmatch()
    if do_Schlaufman_14:
        make_Schalufman14_TIC_crossmatch()

    if find_which_alerts_are_interesting:

        if not find_alerts_in_MWSC:
            print(50*'#')
            print('WARNING: ignoring MWSC lists!')
            print(50*'#')

        crossmatch_alerts(
            '../data/sector_0_TOI_list.txt',
            find_alerts_in_MWSC=find_alerts_in_MWSC
        )

    if make_diagnostic_plots:
        t = ascii.read(
            '../results/Gagne_2018_BANYAN_XI_TIC_crossmatched_10arcsec_maxsep.csv')
        plot_xmatch_separations(t,
            '../results/Gagne18_BANYAN_XI_TIC_crossmatch_separations.png')
        make_Gagne18_skymaps(t)

    if make_diagnostic_plots:
        mwsc_xmatch_dir = '../results/MWSC_TIC_crossmatched/'
        fnames = glob(mwsc_xmatch_dir+
                      '01?[0-1]_*_1sigma_members_TIC_crossmatched.csv')
        for fname in fnames:
            t = ascii.read(fname)
            plot_xmatch_separations(
                t, fname.replace('.csv','_separationhist.png'))

