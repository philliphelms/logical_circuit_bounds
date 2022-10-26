# Clock Device Calculations
## Authors: Phillip Helms & David Limmer
This directory contains the code and scripts for running the calculations
whose results are shown for the clock device in the paper
"Stochastic thermodynamic bounds on logical circuit operation"

## Autocorrelation Function Calculations
To run the autocorrelation calculations shown in Figure 3(a), 
you can run the following from this directory
```
Vd=5
python scripts/calc_acf.py $Vd
```
which will save the resulting data in 
the file:
```
./data/acf_data_Vd{Vd}_Cg10.0.npz
```

## Example trajectories
To generate example trajectories, such as those shown in 
Figure 3(a), you can run the following from this director:
```
Vd=5
python scripts/example_trajectories.py $Vd
```
which will save the resulting trajectory as an image in 
```
./data/example_trajectory_Cg10.0_Vd{Vd}.png
```

## Thermodynamic Uncertainty Relation Calculations
To run calculations for the thermodynamic uncertainty 
relation comparison shown in Figures 2(b-c), you can run the following
from this directory:
```
Vd=5
python scripts/calc_tur.py $Vd
python scripts/calc_controlled_tur.py $Vd
```
which will respectively save the results as
```
./data/tur_data_Vd{Vd}_Cg10.0.npz
./data/controlled_tur_data_Vd{Vd}_Cg10.0.npz
```
