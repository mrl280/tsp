"""
Find the cycle costs from the input cost matrix using pre-computed Hamiltonian cycles.
"""

import numpy as np
from numpy.typing import ArrayLike

from pre_computed_hamiltonian_cycles import pre_computed_hamiltonian_cycles


def find_cycle_costs(A: ArrayLike) -> ArrayLike:
    """
    Given a cost matrix A, find the total cost of each Hamiltonian cycle (Hamiltonian cycles are precomputed)

    :param A:
        A square cost matrix, where <i|A|j> is the cost to travel from city i to city j where 0 < i, j < len(A).
        All elements must be >= 0.

    :return: np.array:
        A 1-d array of len((n - 1)!) containing cycle costs
    """
    try:
        A = np.asarray(A)
    except BaseException as e:
        raise Exception("Error: Unable to convert input A to array. " + str(e))

    # Check to make sure A is square.
    if A.shape[0] != A.shape[1]:
        raise Exception("Error: Cost matrix A must be square!")

    # Check to make sure there are no negative costs.
    if np.any(A < 0.0):
        raise Exception("Error: The cost matrix A cannot contain negative values.")

    # Get the maximum finite value.
    max_cost = np.amax(A[(A != np.inf) & (A != np.nan)])
    n = len(A)  # This is the number of cities in our problem.

    # Remove all infinities and nans.
    A = np.nan_to_num(A, copy=True, nan=0.0, posinf=n * max_cost)

    # Use the precomputed Hamiltonian cycles to compute the costs.
    hamiltonian_cycles = pre_computed_hamiltonian_cycles(n=n)
    number_of_cycles = len(hamiltonian_cycles)

    cycle_costs = np.full(shape=number_of_cycles, fill_value=0.0)  # Pre-allocate.
    for i, cycle in enumerate(hamiltonian_cycles):
        # Each cycle is a list of tuples.
        for t in cycle:
            cycle_costs[i] += A[t[0], t[1]]

    return cycle_costs


if __name__ == "__main__":

    # print("n = 3:")
    # A3 = np.asarray([[3, 7, 2],
    #                  [5, 12, 9],
    #                  [17, 1, np.inf]])
    #
    # print("\nA:")
    # print(A3)
    #
    # print("\ncosts:")
    # costs = find_cycle_costs(A=A3)
    # print(costs)
    #
    # print("\nn = 4:")
    # A4 = np.asarray([[3, 7, 2, 19],
    #                  [5, 12, np.nan, 5],
    #                  [17, 1, np.inf, 2.4],
    #                  [3, 13, 13, 12]])
    #
    # print("\nA:")
    # print(A4)
    #
    # print("\ncosts:")
    # costs = find_cycle_costs(A=A4)
    # print(costs)

    # Poster Example
    print("\nPoster Example, n = 4:")
    A = np.asarray([[0, 7, 2, 7],
                    [5, 0, 3, 5],
                    [2, 1, 0, 8],
                    [3, 13, 6, 0]])

    print("\nA:")
    print(A)

    print("\ncosts:")
    costs = find_cycle_costs(A=A)
    print(costs)
