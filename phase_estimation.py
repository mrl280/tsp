"""
Here, we investigate using phase estimation to obtain the cycle costs.

Notice this is something of an academic exercise to gain an improved understanding of phase estimation, as our problem
 formulation allowed us to compute the cycle costs directly from the cost matrix and pre-computed Hamiltonian cycles.
 We assumed that if we knew the Hamiltonian cycles, we might as well use them to obtain the costs directly.

However, there are other approaches, such as the one presented in Srinivasan et. al.
 (https://arxiv.org/abs/1805.10928), where they only use the Hamiltonian cycles to gain insight into which of their n^n
 eigenvalues correspond to the (n - 1)! Hamiltonian cycles. As a consequence, they have to perform phase estimation with
 a much larger U matrix then we will here.
"""
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def build_U(cycle_costs: ArrayLike) -> ArrayLike:
    """
    Build a unitary matrix U, wherein the costs are encoded as phases.

    :param cycle_costs:
        A 1-d array of len((n - 1)!) (where n is the number of cities) containing cycle costs.
    :return: , tuple:
        np.array: 2-dimensional unitary matrix U.
        tuple: cycle cost range
    """
    # Normalize all the cycle costs so they are between pi / 2 and 3 * pi / 2 (We don't use the full 0 -> 2 * pi range
    #  because numerical errors enable wrapping).
    # Notice we save the min and max so, later on, we can recover the costs from the phases.
    cycle_costs_min = np.amin(cycle_costs)
    cycle_costs -= cycle_costs_min + np.pi / 2
    cycle_costs_max = np.amax(cycle_costs)
    cycle_costs *= (3 * np.pi / 2) / np.amax(cycle_costs)

    print("\ncycle costs after normalization:")
    print(cycle_costs)

    U = np.diag(v=cycle_costs, k=0)
    U = 1j * U
    U = np.exp(U)

    return U, (cycle_costs_min, cycle_costs_max)


def costs_by_phase_estimation(U: ArrayLike, cycle_cost_range: Tuple[int, int]):
    """

    :param U:
    :param cycle_cost_range:
    :return:
    """
    # Obtain the cycle costs from U by with phase estimation.
    number_of_cycles = len(U)  # All eigenstates of U correspond to Hamiltonian cycles.

    cycle_costs = np.fill(shape=number_of_cycles, fill_value=0.0)

    for i in range(number_of_cycles):
        # Perform phase estimation to solve U|i> = lambda |i>
        a = 1  # TODO


    # Now that we have the eigenvalue phases, convert them back into costs.
    cycle_costs *= cycle_cost_range[1] / (3 * np.pi / 2)
    cycle_costs += cycle_cost_range[0] + np.pi / 2

    return cycle_costs


if __name__ == "__main__":
    from find_cycle_costs import find_cycle_costs

    print("n = 3:")
    A3 = np.asarray([[3, 7, 2],
                     [5, 12, 9],
                     [17, 1, np.inf]])

    print("\nA:")
    print(A3)

    print("\nCosts:")
    costs = find_cycle_costs(A=A3)
    print(costs)

    U_, cycle_cost_range_ = build_U(cycle_costs=costs)
    print("\nCycle cost range:")
    print(U_)

    # print("\nn = 4:")
    # A4 = np.asarray([[3, 7, 2, 19],
    #                  [5, 12, np.nan, 5],
    #                  [17, 1, np.inf, 2.4],
    #                  [3, 13, 13, 12]])
    #
    # print("\nA:")
    # print(A4)

