#! /bin/bash
for i in `jot - 1 $1`; do
	(time ode/demo/demo_step_stripped $2 $3) 2>&1 | tr '\n' ' ' | awk '{ print $2 }' | tr 'm|s' ' '
done
