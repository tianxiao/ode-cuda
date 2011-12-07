#! /bin/bash
./iter.sh $1 $2 $3 $4 $5| awk '{ s += $2 } END { print "sum: ", s, " average: ", s/NR, " samples: ", NR }'
