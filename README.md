this repo contains tools and projects related to planets with ages.

this includes:

* a study of whether we could disentangle age from metallicity
  dependence in CKS VII. (we could not. metallicity seems more important.)

* a study of whether multiplanet CKS systems show any changes vs time. e.g.,
  does secular chaos lead to more widely spaced systems? (current status is
  maybe, but the statistics aren't convincing).


```
src/
├── age_plots.py - scatter,histogram,quartile for SD18, exoarchive, & CKS
├── cks_age_exploration.py - do CKS ages affect the planet population?
├── cks_multis_vs_age.py - do CKS ages affect multis specifically?
├── crossmatch_catalogs_vs_TIC.py - tools to xmatch catalogs vs TIC
├── crossmatch_exoarchive_sanders.py - dl, xmatch exoarchive vs SD18
├── download_furlan_2017_results.py
├── exoarchive_age_plots.py - dl exoarchive and make scatter plots
├── get_tess_alert_search_list.py - search catalogs for interesting stars
├── mast_examples.py
├── mast_utils.py - MAST utilities to interface with TIC
├── parse_MWSC.py - legacy code from TESS GI proposal to parse K+13
├── plot_cluster_positions.py
├── plot_exoarchive_sanders18_matches.py
└── select_best_age_plots.sh
```

* `G+18:` Gagne et al, 2018 (associations, moving groups)
* `SD18:` Sanders & Das, 2018 (ages thru GALAH, RAVE, LAMOST, etc.)
* `K+13:` Kharchenko et al, 2013 (MWSC cluster catalog)
* `CKS :` Johnson+I, Petigura+II, Petigura+IV, Weiss+VI, Fulton & Petigura VII.
* `K18 :` Kane, 2018 (known planets)
