#!/bin/bash

pdfpath="data/all_pages.pdf"
outdir="data/images"
prefix="page"

scale=2

width=$((3600 * ${scale}))
height=$((2400 * ${scale}))
blocksize=800
echo $width $height $blocksize

FORMAT="%04g"

halfsize=$((${blocksize} / 2))
pagecnt=$(pdfinfo ${pdfpath} | awk '/^Pages:/ {print $2}')

rm -rf ${outdir}

for page in `seq 1 1 ${pagecnt}`; do
    pagedir=${outdir}/${prefix}-${page}
    mkdir -p ${pagedir}

    for xmax in `seq -w ${blocksize} ${halfsize} ${width}`; do
        xmin=$((10#${xmax} - ${blocksize}))
        printf -v xmin ${FORMAT} ${xmin}

        for ymax in `seq -w ${blocksize} ${halfsize} ${height}`; do
            ymin=$((10#${ymax} - ${blocksize}))
            printf -v ymin ${FORMAT} ${ymin}

            pdftoppm -jpeg -r $((100 * ${scale})) -f ${page} -l ${page} -singlefile \
                -x ${xmin} -y ${ymin} -sz ${blocksize} \
                ${pdfpath} "${pagedir}/${page}__${xmin}@${xmax}__${ymin}@${ymax}"
        done
        
    done
done
