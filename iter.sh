#! /bin/bash
for i in `jot - 1 $1`; do
	(time ode/demo/fat_matrix $2 $3 $4 $5) 2>&1 | tr '\n' ' ' | awk '{ print $2 }' | tr 'm|s' ' '
done
