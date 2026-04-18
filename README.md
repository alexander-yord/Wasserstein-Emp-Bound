# Wasserstein-Emp-Bound

Notes on the AVM folders:
- paths1 and paths2 are the two systems of $E=12J, 120J$ run on my computer
- AVMs is obsolete and incomplete; it used the built-in scipy wasserstein function (generated using: `WassDistr.ipynb`)
- AVMs2 is what we use as the most regular AVMs of the original data (generated using: `WassDistr2.ipynb`)
- AVMs3 is the AVMs from the server simulations
- AVMs4 is the AVMs computed based on every b-th observation (generated using: `WassDistr4.ipynb`)
- AVMs4a is obsolete and incomplete; it used the built-in scipy wasserstein function (generated using: `WassDistr4.ipynb` too)
- AVMs4b contains the correct AVMs calculated on every $b$-th element in the sequence (generated using: `WassDistr4.ipynb` too)
- AVMs5a contains pairwise AVMs calculated using the trapezoidal estimation and a burn-in = 50k observation (generated using: `WassDistr5.ipynb`)
- AVMs5b contains pairwise AVMs calculated using the scipy wasserstein function and a burn-in = 50k observation (generated using: `WassDistr5.ipynb` too)
- AVMs6x is what we use as the most regular AVMs of the original data (generated using: `WassDistr2.ipynb`) but with a burn in = 50k observations
- 
- UAVMs is the AVMs of the PIT-transformed data from our own simulations (generated using: `UPooledWassDistr.ipynb`)

Notes on the Gaussian empirical process folders: 
- `gaussian_empirical_process_output`: original estimates (lag_cutoff = 50; burn-in = 0)
- `gaussian_empirical_process_output1`: estimates (lag_cutoff = 200; burn-in = 5k)
- `gaussian_empirical_process_output2`: original estimates (lag_cutoff = 250; burn-in = 50k) using the parallelized script
- `gaussian_empirical_process_output6`: estimates using the flat-hat kernel (lag_cutoff = 250; burn-in = 50k) and compared with `AVMs6` 



