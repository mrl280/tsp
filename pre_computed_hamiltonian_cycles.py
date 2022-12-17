"""
Author: Michael Luciuk
Date:   Dec, 2022

We assume access to pre-computed Hamiltonian cycles, they are defined herein.
"""
from typing import Tuple


def pre_computed_hamiltonian_cycles(n: int) -> list[list[Tuple]]:
    """
    Return a list of pre-computed Hamiltonian cycles.

    For a cost matrix of size n (n cities) there are (n - 1)! Hamiltonian cycles.

    :param n: int:
        The dimension of the provided cost matrix A (the number of cities in the problem).

    :return: list of lists tuples:
        A list of Hamiltonian cycles.
        Each Hamiltonian cycle is a list of n tuples corresponding to the indices containing the costs for the cycle.
    """

    if n == 2:
        # 2 cities; (2 - 1)! = 1 Hamiltonian cycle.
        return [[(0, 1), (1, 0)]]  # 1 -> 2, 2 -> 1

    if n == 3:
        # 3 cities; (3 - 1)! = 2 Hamiltonian cycles.

        return [[(0, 1), (1, 2), (2, 0)],  # 1 -> 2, 2 -> 3, 3 -> 1
                [(0, 2), (1, 0), (2, 1)]]  # 1 -> 3, 3 -> 2, 2 -> 1

    if n == 4:
        # 4 cities; (4 - 1)! = 6 Hamiltonian cycles.

        return [[(0, 1), (1, 2), (2, 3), (3, 0)],  # 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 1
                [(0, 1), (1, 3), (3, 2), (2, 0)],  # 1 -> 2, 2 -> 4, 4 -> 3, 3 -> 1
                [(0, 2), (2, 1), (1, 3), (3, 0)],  # 1 -> 3, 3 -> 2, 2 -> 4, 4 -> 1
                [(0, 2), (2, 3), (3, 1), (1, 0)],  # 1 -> 3, 3 -> 4, 4 -> 2, 2 -> 1
                [(0, 3), (3, 1), (1, 2), (2, 0)],  # 1 -> 4, 4 -> 2, 2 -> 3, 3 -> 1
                [(0, 3), (3, 2), (2, 1), (1, 0)]]  # 1 -> 4, 4 -> 3, 3 -> 2, 2 -> 1

    else:
        raise NotImplementedError
