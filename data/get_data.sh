#!/bin/bash

mkdir cgos
base_url="http://www.yss-aya.com/cgos/9x9/archives"

for year in {2015..2025}; do
    for month in {01..12}; do
        file="9x9_${year}_${month}.tar.bz2"
        url="${base_url}/${file}"

        echo "Downloading ${url} if it exists..."
        wget --no-clobber --continue --retry-connrefused --timeout=10 --tries=3 -P cgos "${url}" || echo "File ${file} missing, skipping..."
        echo "Extracting $file..."
        tar -xf cgos/$file --directory cgos
        echo "Removing $file..."
        rm -r cgos/$file
    done
done