# Memory Device Calculations
## Authors: Phillip Helms & David Limmer
This directory contains the code and scripts for running the calculations
whose results are shown for the memory device in the paper
"Stochastic thermodynamic bounds on logical circuit operation"

## Steady State Phase Diagram
To get the steady state distribution, used to construct the 
phase diagram in Figure 2(a), you can run the following 
from this directory
```
Vd=5
Cg=10
dt=0.1
python scripts/steady_state.py $Cg $Vd $dt
```
which will run the calculations and save the resulting data as
```
/data/steady_state_Cg10.0_Vd5.0_dt0.1_steady_initial_state.npz
```
Repeating the calculations with many values of Vd will 
produce the data needed for Figure 2(a).

## Example Trajectories
You can run the following:
```
python scripts/kmc.py
```
to generate an example trajectory, as are included in Figure 2(a). 
By varying lines 8-10 in the script, you can control the cross-voltage,
gate capacitance, and overall simulation time. 

This file will save a figure at:
```
./data/example_evolution.png
```

## First Passage Time Calculations
To run the first passage time calculations, you can run the following
from this directory
```
Vd=5
Cg=10
dt=0.1
python scripts/calc_fpt.py $Cg $Vd $dt
```
which will run the calculations and save the resulting data as
```
/data/fpt_Cg10.0_Vd5.0_dt0.1_steady_initial_state.npz
```
Repeating the calculations with many values of Vd will 
produce the data needed for Figure 2(b).
