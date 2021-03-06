# Compare LIRS and other cache replacement policies(LRU, SLRU, FIFO, RAND)

This example runs experiments using different combinations of topologies,
caching strategies, content popularity distributions, cache sizes and cache policies and plot
the results on a number of graphs.

## Run
To run the experiments and plot the results, execute:

    $ make

## How does it work
The `config.py` contains all the configuration for executing experiments and
do plots. Running `make` launches the Icarus simulator passing the configuration
file as an argument and plots results. The `plotresults.py` file provides functions
for plotting specific results based on `icarus.results.plot` functions.
