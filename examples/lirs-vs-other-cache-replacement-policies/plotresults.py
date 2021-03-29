#!/usr/bin/env python
"""Plot results read from a result set
"""
from __future__ import division
import os
import argparse
import logging

import matplotlib.pyplot as plt

from icarus.util import Settings, config_logging
from icarus.results import plot_lines, plot_bar_chart
from icarus.registry import RESULTS_READER


# Logger object
logger = logging.getLogger('plot')

# These lines prevent insertion of Type 3 fonts in figures
# Publishers don't want them
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True

# If True text is interpreted as LaTeX, e.g. underscore are interpreted as
# subscript. If False, text is interpreted literally
plt.rcParams['text.usetex'] = False

# Aspect ratio of the output figures
plt.rcParams['figure.figsize'] = 8, 5

# Size of font in legends
LEGEND_SIZE = 14

# Line width in pixels
LINE_WIDTH = 1.5

# Plot
PLOT_EMPTY_GRAPHS = True

# This dict maps policy names to the style of the line to be used in the plots
POLICY_STYLE = {          
         'LRU':             'b--p',
         'LIRS':            'g-->',
         'ARC':             'y-d',
         'SLRU':            'r-o',
         'IN_CACHE_LFU':    'k-.^',
         'FIFO':            'c--<',
         'RAND':            'm--*'
                }

# This dict maps name of policies to names to be displayed in the legend
POLICY_LEGEND = {
         'LRU':             'LRU',
         'LIRS':            'LIRS',
         'ARC':             'ARC', 
         'SLRU':            'SLRU',
         'IN_CACHE_LFU':    'IN_CACHE_LFU',
         'FIFO':            'FIFO',
         'RAND':            'RAND'     
                    }

# Color and hatch styles for bar charts of cache hit ratio and link load vs topology
POLICY_BAR_COLOR = {
    'LRU':           'k',
    'LIRS':          '0.4',
    'SLRU':          '0.5',
    'FIFO':          '0.6',
    'RAND':          '0.7',
    'ARC':           '0.8',
    'IN_CACHE_LFU':  '0.9'
    }

POLICY_BAR_HATCH = {
    'LRU':          None,
    'LIRS':         '//',
    'SLRU':         'x',
    'FIFO':         '+',
    'RAND':         '\\',
    'ARC':          'o',
    'IN_CACHE_LFU': '|'
    }




def plot_cache_hits_vs_cache_size(resultset, topology, strategy, cache_size_range, policies, plotdir):
    desc = {}
    desc['title'] = 'Cache hit ratio: T=%s S=%s' % (topology, strategy)
    desc['xlabel'] = u'Cache to population ratio'
    desc['ylabel'] = 'Cache hit ratio'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement', 'network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'strategy': {'name': strategy}}
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')] * len(policies)
    desc['ycondnames'] = [('cache_policy', 'name')] * len(policies)
    desc['ycondvals'] = policies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper left'
    desc['line_style'] = POLICY_STYLE
    desc['legend'] = POLICY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'CACHE_HIT_RATIO_T=%s & S=%s.pdf'
               % (topology, strategy), plotdir)



def plot_link_load_vs_cache_size(resultset, topology, strategy, cache_size_range, policies, plotdir):
    desc = {}
    desc['title'] = 'Internal link load: T=%s S=%s' % (topology, strategy)
    desc['xlabel'] = 'Cache to population ratio'
    desc['ylabel'] = 'Internal link load'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement', 'network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'strategy': {'name': strategy}}
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')] * len(policies)
    desc['ycondnames'] = [('cache_policy', 'name')] * len(policies)
    desc['ycondvals'] = policies
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = POLICY_STYLE
    desc['legend'] = POLICY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LINK_LOAD_INTERNAL_T=%s & S=%s.pdf'
               % (topology, strategy), plotdir)


def plot_latency_vs_cache_size(resultset, topology, strategy, cache_size_range, policies, plotdir):
    desc = {}
    desc['title'] = 'Latency: T=%s S=%s' % (topology, strategy)
    desc['xlabel'] = 'Cache to population ratio'
    desc['ylabel'] = 'Latency'
    desc['xscale'] = 'log'
    desc['xparam'] = ('cache_placement', 'network_cache')
    desc['xvals'] = cache_size_range
    desc['filter'] = {'topology': {'name': topology},
                      'strategy': {'name': strategy}}
    desc['ymetrics'] = [('LATENCY', 'MEAN')] * len(policies)
    desc['ycondnames'] = [('cache_policy', 'name')] * len(policies)
    desc['ycondvals'] = policies
    desc['metric'] = ('LATENCY', 'MEAN')
    desc['errorbar'] = True
    desc['legend_loc'] = 'upper right'
    desc['line_style'] = POLICY_STYLE
    desc['legend'] = POLICY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_lines(resultset, desc, 'LATENCY_T=%s & S=%s.pdf'
               % (topology, strategy), plotdir)


def plot_cache_hits_vs_topology(resultset, strategy, cache_size, topology_range, policies, plotdir):
    desc = {}
    desc['title'] = 'Cache hit ratio: S=%s C=%s' % (strategy, cache_size)
    desc['ylabel'] = 'Cache hit ratio'
    desc['xparam'] = ('topology', 'name')
    desc['xvals'] = topology_range
    desc['filter'] = {'strategy': {'name': strategy},
                      'cache_placement': {'network_cache': cache_size}}
    desc['ymetrics'] = [('CACHE_HIT_RATIO', 'MEAN')] * len(policies)
    desc['ycondnames'] = [('cache_policy', 'name')] * len(policies)
    desc['ycondvals'] = policies
    desc['errorbar'] = True
    desc['legend_loc'] = 'lower right'
    desc['bar_color'] = POLICY_BAR_COLOR
    desc['bar_hatch'] = POLICY_BAR_HATCH
    desc['legend'] = POLICY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(resultset, desc, 'CACHE_HIT_RATIO_S=%s & C=%s.pdf'
                   % (strategy, cache_size), plotdir)


def plot_link_load_vs_topology(resultset, strategy, cache_size, topology_range, policies, plotdir):
    desc = {}
    desc['title'] = 'Internal link load: A=%s C=%s' % (strategy, cache_size)
    desc['ylabel'] = 'Internal link load'
    desc['xparam'] = ('topology', 'name')
    desc['xvals'] = topology_range
    desc['filter'] = {'strategy': {'name': strategy},
                      'cache_placement': {'network_cache': cache_size}}
    desc['ymetrics'] = [('LINK_LOAD', 'MEAN_INTERNAL')] * len(policies)
    desc['ycondnames'] = [('cache_policy', 'name')] * len(policies)
    desc['ycondvals'] = policies
    desc['errorbar'] = True
    desc['legend_loc'] = 'lower right'
    desc['bar_color'] = POLICY_BAR_COLOR
    desc['bar_hatch'] = POLICY_BAR_HATCH
    desc['legend'] = POLICY_LEGEND
    desc['plotempty'] = PLOT_EMPTY_GRAPHS
    plot_bar_chart(resultset, desc, 'LINK_LOAD_INTERNAL_S=%s & C=%s.pdf'
                   % (strategy, cache_size), plotdir)




def run(config, results, plotdir):
    """Run the plot script

    Parameters
    ----------
    config : str
        The path of the configuration file
    results : str
        The file storing the experiment results
    plotdir : str
        The directory into which graphs will be saved
    """
    settings = Settings()
    settings.read_from(config)
    config_logging(settings.LOG_LEVEL)
    resultset = RESULTS_READER[settings.RESULTS_FORMAT](results)
    # Create dir if not existsing
    if not os.path.exists(plotdir):
        os.makedirs(plotdir)
    # Parse params from settings
    alphas = settings.ALPHA
    topologies = settings.TOPOLOGIES
    cache_sizes = settings.NETWORK_CACHE
    strategies = settings.STRATEGIES
    policies = settings.CACHE_POLICIES
    # Plot graphs
    for topology in topologies:
        for strategy in strategies:
            logger.info('Plotting cache hit ratio for topology %s and strategy %s vs cache size' % (topology, strategy))
            plot_cache_hits_vs_cache_size(resultset, topology, strategy, cache_sizes, policies, plotdir)
            logger.info('Plotting link load for topology %s and strategy %s vs cache size' % (topology, strategy))
            plot_link_load_vs_cache_size(resultset, topology, strategy, cache_sizes, policies, plotdir)
            logger.info('Plotting latency for topology %s and strategy %s vs cache size' % (topology, strategy))
            plot_latency_vs_cache_size(resultset, topology, strategy, cache_sizes, policies, plotdir)
    for cache_size in cache_sizes:
        for strategy in strategies:
            logger.info('Plotting cache hit ratio for cache size %s vs strategy %s against topologies' % (str(cache_size), strategy))
            plot_cache_hits_vs_topology(resultset, strategy, cache_size, topologies, policies, plotdir)
            logger.info('Plotting link load for cache size %s vs strategy %s against topologies' % (str(cache_size), strategy))
            plot_link_load_vs_topology(resultset, strategy, cache_size, topologies, policies, plotdir)
    logger.info('Exit. Plots were saved in directory %s' % os.path.abspath(plotdir))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-r", "--results", dest="results",
                        help='the results file',
                        required=True)
    parser.add_argument("-o", "--output", dest="output",
                        help='the output directory where plots will be saved',
                        required=True)
    parser.add_argument("config",
                        help="the configuration file")
    args = parser.parse_args()
    run(args.config, args.results, args.output)


if __name__ == '__main__':
    main()
