"""
Author: Michael Luciuk
Date:   Dec, 2022

Stragety based on:
 K. Srinivasan, S. Satyajit, B. K. Behera, and P. K. Panigrahi, “Efficient quantum algorithm for solving travelling
  salesman problem: An IBM quantum experience,” 2018, doi: 10.48550/arxiv.1805.10928.
"""
import numpy as np

from grover_enhanced_minimization import grover_enhanced_minimization
from phase_estimation import build_U, phase_estimation

print("3 City Example (n = 3):")
A = np.asarray([[3, 7, 2],
                [5, 12, 9],
                [17, 1, 6]])

print("\nA:")
print(A)

# Using A, build a unitary matrix where the costs are encoded as phases.
U, cycle_cost_range = build_U(A=A, verbose=False)

# Use phase estimation to obtain the phases (phases encode total costs).
thetas = phase_estimation(U=U)

# Now that we have the phases, convert them back into costs.
cycle_costs_found = thetas * cycle_cost_range[1]
cycle_costs_found += cycle_cost_range[0]

print("\nTotal costs:")
print(cycle_costs_found)

# Right now our Grover enhanced minimization only works for interger lists. Our toal costs should be integer anyway.
cycle_costs_rounded = [int(item) for item in cycle_costs_found]
lowest_cost = grover_enhanced_minimization(arr=cycle_costs_rounded, verbose=False)

print("\nShortest cost:")
print(lowest_cost)
