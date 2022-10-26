# Logical NOT Gate Calculations
## Authors: Phillip Helms & David Limmer
This directory contains the code and scripts for running the calculations
whose results are shown for the single logical NOT gate in the paper
"Stochastic thermodynamic bounds on logical circuit operation"

## First Passage Time Calculations
To run the first passage time calculations, you can run the following
from this directory
```
python scripts/charging_first_passage_time.py
python scripts/discharging_first_passage_time.py
```
which respectively run calcualtions for the charging and 
discharging operation of the logical NOT gate. 

To change the cross-voltage, accuracy threshold, and gate capacitance, 
you should alter lines 16-18 of these scripts. 

The resulting data will be stored as:
```
./data/first_passage_time_distribution_forward_alpha{alpha}_Cg{Cg}.npz
```
containing a sweep over many cross-voltages for a single accuracy
threshold and gate capacitance. 

## Steady State Calculations
To run calculations to find the steady state output probability
distribution for a single NOT gate, you can run the following
from this directory:
```
python scripts/steady_state_accuracy.py
```
and change lines 17-19 to alter the cross-voltage, accuracy thresholds, 
and gate capacitances. 

The resulting data will be stored as:
```
./data/steady_state_discharging.npz
```

## Time Evolution Calculation
To perform the time evolution shown in Figure 1(a), 
you can run the following script from this 
directory:
```
python scripts/time_evolution.py
```
which will then save the resulting plot at
```
./data/time_evolution.pdf
```
