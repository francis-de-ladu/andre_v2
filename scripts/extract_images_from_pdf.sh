#!/bin/bash

pdfpath="data/all_pages.pdf"
outdir="data/images"
prefix="page"

width=3600
height=2400
blocksize=600


pagecnt=$(pdfinfo ${pdfpath} | awk '/^Pages:/ {print $2}')

for page in $(seq 1 1 ${pagecnt}); do
    pagedir=${outdir}/${prefix}-${page}
    mkdir ${pagedir}

    for xmax in $(seq ${blocksize} ${blocksize} ${width}); do
        xmin=$((${xmax} - ${blocksize}))
        # xi=$((${x}/${blocksize}))
        for ymax in $(seq ${blocksize} ${blocksize} ${height}); do
            ymin=$((${ymax} - ${blocksize}))
            # yi=$((${y}/${blocksize}))
            pdftoppm -jpeg -r 100 -f ${page} -l ${page} -singlefile \
                -x ${xmin} -y ${ymin} -sz ${blocksize} \
                ${pdfpath} "${pagedir}/${xmin}@${xmax}__${ymin}@${ymax}"
        done
    done
done
