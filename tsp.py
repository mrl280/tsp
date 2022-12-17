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


def tsp(A: list[list[int]]) -> int:
    """
    A "quantum-enhanced" TSP solver. Uses a variation of the algorithm presented by Srinivasan and collegues in
     “Efficient quantum algorithm for solving travelling salesman problem: An IBM quantum experience.”

    # TODO: Return the lowest cost circuit, rather than just the cost of the lowest-cost circuit.

    :param A: 2d list of ints:
        A square cost matrix, where <i|A|j> is the cost to travel from city i to city j where 0 < i, j < len(A).
        All elements must be >= 0.

    :return: int:
        The cost of the lowest-cost Hamiltonian cycle.
    """
    # Using A, build a unitary matrix where the costs are encoded as phases.
    U, cycle_cost_range = build_U(A=A, verbose=False)

    # Use phase estimation to obtain the phases (phases encode total costs).
    thetas = phase_estimation(U=U)

    # Now that we have the phases, convert them back into costs.
    cycle_costs_found = thetas * cycle_cost_range[1]
    cycle_costs_found += cycle_cost_range[0]

    # Right now our Grover enhanced minimization only works for interger lists. Our toal costs should be integer anyway.
    cycle_costs_rounded = [int(item) for item in cycle_costs_found]
    lowest_cost = grover_enhanced_minimization(arr=cycle_costs_rounded, verbose=False)

    return lowest_cost


if __name__ == "__main__":

    print("### 3 City Example (n = 3) ###")
    A = [[0, 7, 2],
         [5, 0, 9],
         [17, 1, 0]]

    print("\nA:")
    print(A)

    print("\nCost of shortest cost circuit:")
    print(tsp(A=A))

    print("\n### 4 city Example (n = 4) ### ")
    A = [[3, 7, 2, 19],
         [5, 12, np.nan, 5],
         [17, 1, np.inf, 2.4],
         [3, 13, 13, 12]]

    print("\nA:")
    print(A)

    print("\nCost of shortest cost circuit:")
    print(tsp(A=A))
