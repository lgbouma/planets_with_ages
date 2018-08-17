#!/usr/bin/env bash

# PURPOSE
# copy the exoplanet-archive ages plots to a separate directory,
# `results/best_exoarchive_age_plots`

indir=../results/exoarchive_age_plots
outdir=../results/best_exoarchive_age_plots

echo "copying best exoarchive plots"
cp $indir/age_vs_log_pl_orbper.png $outdir/.
cp $indir/log_age_vs_log_pl_rade.png $outdir/.
cp $indir/log_age_vs_st_glat.png $outdir/.
cp $indir/age_vs_log_pl_rade.png $outdir/.
cp $indir/age_vs_pl_rade.png $outdir/.
cp $indir/age_vs_pl_pnum.png $outdir/.
cp $indir/log_age_vs_pl_orbincl.png $outdir/.
cp $indir/age_vs_pl_dens.png $outdir/.
cp $indir/age_vs_pl_orbincl.png $outdir/.

indir=../results/sd18_age_plots
outdir=../results/best_sd18_age_plots

echo "copying best Sanders Das 2018 plots"
cp $indir/age_vs_log_pl_orbper.png $outdir/.
cp $indir/log_age_vs_log_pl_rade.png $outdir/.
cp $indir/log_age_vs_st_glat.png $outdir/.
cp $indir/age_vs_log_pl_rade.png $outdir/.
cp $indir/age_vs_pl_rade.png $outdir/.
cp $indir/age_vs_pl_pnum.png $outdir/.
cp $indir/log_age_vs_pl_orbincl.png $outdir/.
cp $indir/age_vs_pl_dens.png $outdir/.
cp $indir/age_vs_pl_orbincl.png $outdir/.

indir=../results/cks_age_plots
outdir=../results/best_cks_age_plots

echo "copying best CKS plots"
cp $indir/age_vs_log_iso_prad.png        $outdir/.
cp $indir/age_vs_log_koi_period.png      $outdir/.
cp $indir/log_age_vs_cks_smet_VII.png    $outdir/.
cp $indir/log_age_vs_log_iso_prad.png    $outdir/.
cp $indir/log_age_vs_log_koi_period.png  $outdir/.
cp $indir/age_vs_log_koi_dor.png         $outdir/.

