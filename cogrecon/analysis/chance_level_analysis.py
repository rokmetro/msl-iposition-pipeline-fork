from __future__ import print_function
from fractions import Fraction
# noinspection PyUnresolvedReferences
import numpy as np
import itertools
import cogrecon.core.full_pipeline as pipe
# noinspection PyUnresolvedReferences
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from matplotlib_venn import venn3, venn3_circles

num_items = 7
item_list = range(1, num_items+1)
perms = list(itertools.permutations(item_list))
# print(perms)
swap_counts = [0]*(num_items+1)
cycle_counts = [0]*(num_items+1)
cycle_sizes = [0]*(num_items+1)
swap_and_cycle = [0]*(num_items+1)
dummy_locs = [[0, 0]]*num_items
num_swaps = 0
num_cycles = 0
for p in perms:
    # noinspection PyRedeclaration
    _, _, ts, ps, tc, pc, components, _, _, _, _, _, _, _, _, _, _, _, _ = pipe.trial_swaps(dummy_locs, dummy_locs,
                                                                                            item_list, list(p),
                                                                                            [True]*num_items, -1)
    swaps = ts+ps
    cycles = tc+pc
    num_swaps += swaps
    num_cycles += cycles
    if swaps > 0 and cycles > 0:
        swap_and_cycle[1] += 1
    swap_counts[swaps] += 1
    cycle_counts[cycles] += 1
    for c in components:
        if len(c) >= 2:
            cycle_sizes[len(c)] += 1

print ("Swap Change: {0}".format(float(num_swaps) / float(len(perms))))
print ("Cycle Chance: {0}".format(float(num_cycles) / float(len(perms))))

print ("Counts")
print (swap_counts)
print (cycle_counts)
print (cycle_sizes)
print (swap_and_cycle)

swap_probabilities = [float(count)/float(len(perms)) for count in swap_counts]
cycle_probabilities = [float(count)/float(len(perms)) for count in cycle_counts]
cycle_size_probabilities = [float(count)/float(len(perms)) for count in cycle_sizes]
swap_and_cycle_probabilities = [float(count)/float(len(perms)) for count in swap_and_cycle]

print ("Fractions")
print ([str(Fraction(x).limit_denominator()) for x in swap_probabilities])
print ([str(Fraction(x).limit_denominator()) for x in cycle_probabilities])
print ([str(Fraction(x).limit_denominator()) for x in cycle_size_probabilities])
print (sum(cycle_size_probabilities))
print ([str(Fraction(x).limit_denominator()) for x in swap_and_cycle_probabilities])

print ("Floats")
print (swap_probabilities)
print (cycle_probabilities)
print (cycle_size_probabilities)
print (sum(cycle_size_probabilities))
print (swap_and_cycle_probabilities)

# venn3(subsets=(1, 1, 1, 2, 1, 2, 2), set_labels=('Set1', 'Set2', 'Set3'))
# plt.show()
