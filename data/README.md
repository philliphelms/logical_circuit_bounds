# Generating Figures 1-3
The files contained in this directory will reproduce each of the figures
in the paper:

Stochastic thermodynamic bounds on logical circuit operation

## Generating Figures
To generate all of the figures, you can run the following:
```
cd not_gate
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
./not_gate/figs/
./memory_device/figs/
./clock/figs/
```

## Data
For all plots, the data displayed in the paper is stored in the 
directories:
```
./not_gate/data/
./memory_device/data/
./clock/data/
```
except for the data displayed in Figure 1 (b), which is generated
when the script is run. 
