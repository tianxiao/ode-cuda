#! /bin/bash
for i in `jot - 1 $3`; do
	(time ode/demo/fat_matrix $1 $2) 2>&1 | tr '\n' ' ' | awk '{ print $2 }' | tr 'm|s' ' '
done
