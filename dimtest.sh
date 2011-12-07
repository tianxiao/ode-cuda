#! /bin/bash
# $1 'c' or 'o'
# $2 samples
# $3 divisions
# $4 start
# $5 p
# $6 q
# $7 r
for P in `jot $3 $4 $5`; do
	for Q in `jot $3 $4 $6`; do
		for R in `jot $3 $4 $7`; do
			echo -e "P: ${P}\tQ: ${Q}\tR: ${R}\t"
			./iter.sh $2 $1 $P $Q $R | awk '{ s += $2 } END { print "SUM: ", s, "\tAVG: ", s/NR, "\tSAMPLES: ", NR }'
		done
	done
done
