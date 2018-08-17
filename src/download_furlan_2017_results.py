import numpy as np, pandas as pd, os
from astropy.table import Table, join
from astropy import units as u, constants as c

from astroquery.vizier import Vizier

def download_furlan_radius_correction_table():
    # Furlan et al 2017

    catalog_list = Vizier.find_catalogs('J/AJ/153/71')

    print({k:v.description for k,v in catalog_list.items()})

    Vizier.ROW_LIMIT = -1

    f17_tablelist = Vizier.get_catalogs(catalog_list.keys())

    companiontab = f17_tablelist[-1]

    df = companiontab.to_pandas()

    for colname in ['KOI','Nc','Nobs']:
        df[colname] = df[colname].astype(int)

    # Assume planet orbits primary; this is lower bound on contamination.
    # Therefore it's weaker than assuming the planet orbits the next-brightest
    # companion. However, it's what Petigura+ 2018 did. (Probably b/c the converse
    # would throw out too many).
    df = df[df['Orbit']==b'primary']

    outname = '../data/Furlan_2017_table9.csv'
    df.to_csv(outname, index=False)
    print('saved to {:s}'.format(outname))
