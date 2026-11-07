#!/usr/bin/env bash
# Download all raw inputs into ../data. Requires network access to:
#   mba.tuck.dartmouth.edu, query1.finance.yahoo.com, raw.githubusercontent.com, pages.stern.nyu.edu
set -euo pipefail
cd "$(dirname "$0")/../data"
UA='Mozilla/5.0'

# Ken French Data Library (US market + 1m T-bill; Developed-markets market return)
for f in F-F_Research_Data_Factors_CSV Developed_3_Factors_CSV; do
  curl -sSf -A "$UA" -o "$f.zip" "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/$f.zip"
  unzip -o -q "$f.zip" && rm "$f.zip"
done

# Yahoo Finance daily bars incl. dividend-adjusted close (ETFs) and yield indices (^FVX, ^IRX)
for t in VT GLD GLDM VGIT IEI SGOV BIL VTI ^FVX ^IRX; do
  curl -sSf -A "$UA" -o "yh_${t#^}.json" \
    "https://query1.finance.yahoo.com/v8/finance/chart/$t?period1=0&period2=1900000000&interval=1d&events=div,split"
done

# Monthly gold price (USD/oz) and BLS CPI-U (NSA), mirrored by datahub.io on GitHub
curl -sSf -o gold_monthly.csv https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv
curl -sSf -o cpi_us.csv https://raw.githubusercontent.com/datasets/cpi-us/main/data/cpiai.csv

# Damodaran annual returns (cross-validation of gold and inflation)
curl -sSf -A "$UA" -o histretSP.xls https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls
echo "done: $(ls | wc -l) files"
