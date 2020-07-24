import matplotlib.pyplot as plt, pandas as pd, numpy as np
import os

from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as c
from astropy.io.votable import parse

from numpy import array as nparr

from cks_age_exploration import (
    _get_cks_data, _apply_cks_IV_metallicity_study_filters
)

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

from billy.plotting import format_ax

def main_deprecated():

    df = _get_cks_data()
    sel = _apply_cks_IV_metallicity_study_filters(df)
    p17_df = df[sel]

    vot = parse('../data/McQuillan_2014_ApJS_211_24_table1.vot')
    m14 = vot.get_first_table()
    m14_df = m14.to_table().to_pandas()

    p17_df['id_kic'] = p17_df['id_kic'].astype(str)
    m14_df['KIC'] = m14_df['KIC'].astype(str)

    # # no matches. wat.
    # mdf = p17_df.merge(m14_df, how='inner', left_on='id_kic', right_on='KIC')

    q = NasaExoplanetArchive.query_criteria(
        table='koi', select='kepoi_name,ra,dec', order='kepoi_name'
    )
    q_df = q.to_pandas()

    mdf0 = p17_df.merge(
        q_df, how='inner', left_on='id_koicand',
        right_on='kepoi_name'
    ).drop_duplicates('id_koicand', keep='first')

    assert len(mdf0) == len(p17_df)

    c_p17 = SkyCoord(ra=nparr(mdf0.ra)*u.deg, dec=nparr(mdf0.dec)*u.deg)
    c_m14 = SkyCoord(ra=nparr(m14_df._RA)*u.deg, dec=nparr(m14_df._DE)*u.deg)

    idx_m14, idx_p17, d2d, _ = c_p17.search_around_sky(c_m14, 1*u.arcsec)

    # NOTE: this yields nothing, because McQuillan+2014 explicitly removed KOIs
    # from their sample.


def main():

    df = _get_cks_data()
    sel = _apply_cks_IV_metallicity_study_filters(df)
    p17_df = df[sel]

    vot = parse('../data/McQuillan_2013_ApJ_775L.vot')
    m13 = vot.get_first_table()
    m13_df = m13.to_table().to_pandas()

    p17_df['id_kic'] = p17_df['id_kic'].astype(str)
    m13_df['KIC'] = m13_df['KIC'].astype(str)

    mdf = p17_df.merge(m13_df, how='inner', left_on='id_kic', right_on='KIC')

    for scale in ['linear','log']:
        f, ax = plt.subplots()
        ax.scatter(mdf.giso_prad, mdf.Prot, s=2, c='k')
        ax.set_xlabel('CKS Planet Size [R$_\oplus$]')
        ax.set_ylabel('M13 Rotation Period [day]')
        ax.set_xscale(scale)
        ax.set_yscale(scale)
        format_ax(ax)
        outpath = f'../results/cks_rotation_period/cks_rp_vs_mcquillan_prot_{scale}.png'
        f.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.close('all')
    f, ax = plt.subplots()
    ax.scatter(mdf.koi_period, mdf.Prot, s=2, c='k')
    ax.set_xlabel('KOI Period [day]')
    ax.set_ylabel('M13 Rotation Period [day]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    format_ax(ax)
    outpath = '../results/cks_rotation_period/koi_period_vs_mcquillan_prot.png'
    f.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.close('all')
    f, ax = plt.subplots()
    ax.scatter(mdf.koi_period, mdf.giso_prad, s=2, c='k')
    ax.set_xlabel('KOI Period [day]')
    ax.set_ylabel('CKS Planet Size [R$_\oplus$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    format_ax(ax)
    outpath = '../results/cks_rotation_period/koi_period_vs_giso_prad_M13_match.png'
    f.savefig(outpath, dpi=300, bbox_inches='tight')




if __name__ == "__main__":
    main()
