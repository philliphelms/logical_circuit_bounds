# Code and Data for "Stochastic thermodynamic bounds on logical circuit operation"
## Authors: Phillip Helms & David Limmer
This repository contains all of the code for the calculations
done in the paper "Stochastic thermodynamic bounds on logical circuit operation", 
as well as the data used in generating the figures. 

## Data & Figures
To generate the figures displayed in the paper, do the following:
### Generating Figures
To generate all of the figures, you can run the following:
```
cd ./data/not_gate
python scripts/fig1b.py
python scripts/fig1c.py
python scripts/fig1d.py
cd ../memory_device
python scripts/fig2a.py
python scripts/fig2b.py
cd ../clock
python scripts/fig3a.py
python scripts/fig3bc.py
```

The resulting figures will be stored in the following directories:
```
./data/not_gate/figs/
./data/memory_device/figs/
./data/clock/figs/
```

### Data
For all plots, the data displayed in the paper is stored in the 
directories:
```
./data/not_gate/data/
./data/memory_device/data/
./data/clock/data/
```
except for the data displayed in Figure 1 (b), which is generated
when the script is run. 

## Running the code and generating data

