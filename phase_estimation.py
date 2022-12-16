"""
Author: Michael Luciuk
Date    Dec, 2022

Here, we investigate using phase estimation to obtain the cycle costs.

Notice this is something of an academic exercise to gain an improved understanding of phase estimation, as our problem
 formulation allowed us to compute the cycle costs directly from the cost matrix and pre-computed Hamiltonian cycles.
 We assumed that if we knew the Hamiltonian cycles, we might as well use them to obtain the costs directly.

However, there are other approaches, such as the one presented in Srinivasan et. al.
 (https://arxiv.org/abs/1805.10928), where they only use the Hamiltonian cycles to gain insight into which of their n^n
 eigenvalues correspond to the (n - 1)! Hamiltonian cycles. As a consequence, they perform phase estimation, and with
 a much larger U matrix then we will here.
"""
import math

import pennylane as qml
from pennylane.templates import QuantumPhaseEstimation
from pennylane import numpy as np

from typing import Tuple


def build_U(A: list[list[int]], verbose: False) -> Tuple[np.asarray, Tuple[float, float]]:
    """
    Build a unitary matrix U, wherein the costs are encoded as phases.

    :param A: 2D list of ints:
        A square cost matrix, where <i|A|j> is the cost to travel from city i to city j where 0 < i, j < len(A).
        All elements must be >= 0.
    :param verbose: bool (optional; default is False):
        Print our extra information - useful for debugging.

    :return:
        np.array: 2-dimensional unitary matrix U.
        tuple: Cycle cost range. Will be eeded to convert from phase back to cost.
    """
    n = len(A)  # Number of cities in the problem
    cycle_costs = np.asarray(find_cycle_costs(A=A), dtype=float)

    # Normalize all the cycle costs so they are in the range [0, 2 * pi).
    # Notice we could use Durr and Hoyer quantum-enhanced optimation here to compute the min and max values of A.
    cycle_costs_max = n * np.amax(A)  # This is the largest possbily cycle cost.
    cycle_costs_min = n * np.amin(A)  # This is the smallest possible cycle cost

    normalized_cycle_costs = cycle_costs - cycle_costs_min
    normalized_cycle_costs *= 2 * np.pi / cycle_costs_max

    diagonal_elements = np.exp(1j * normalized_cycle_costs)
    U = np.diag(v=diagonal_elements, k=0)

    if verbose:
        print("\nCycle costs: " + str(cycle_costs))

        print("Max possible cycle cost: " + str(cycle_costs_max))
        print("Min possible cycle cost: " + str(cycle_costs_min))

        print("\nCycle costs after normalization:")
        print(normalized_cycle_costs)

        print("\nDiagonal elements:")
        print(diagonal_elements)

    return U, (cycle_costs_min, cycle_costs_max)


def phase_estimation(U: list[list]):
    """
    Using phase estimation, estimate the eigenvalues of U.
    Since U is a diagonal matrix of complex exponentials, it must be unitary. Also, its eigenvectors must be the
     standard basis vectors. Also also, the eigenvalues are just the diagonal elements, but let's ignore this.

    :param U: 2d list of complex numbers:
        A diagonal matrix with costs encoded as phases.

    :return: list of floats:
        thetas: A list of estimated phases. These phases represent the total costs.
        From these phases, we could obtain the eignevalues as e^(2pi * i * theta).
    """

    # The number of target wires required depends on the size of U
    n_target_wires = math.ceil(math.log(len(U_), 2))

    # We get to choose the number of estimation wires to use. The error in our estimate will decrease exponentially
    #  with the number of estimation qubits.
    n_estimation_wires = 5

    thetas = np.full(shape=len(U_), fill_value=np.nan)

    for i in range(len(U_)):
        # Perform phase estimation to solve U|i> = lambda |i>

        target_wires = list(range(0, n_target_wires))
        estimation_wires = range(n_target_wires, n_target_wires + n_estimation_wires)

        # Create a device, node, and execute the quntum circuit.
        device = qml.device("default.qubit", wires=n_estimation_wires + 1)
        qnode = qml.qnode(func=phase_estimation_circuit, device=device)()

        values = qnode.__call__(unitary=U, target_wires=target_wires, estimation_wires=estimation_wires, i=i)
        # print("values:")
        # print(values)

        # Find the index of the largest value in the probability distribution and divide that number by 2^n.
        theta = np.argmax(values) / 2 ** n_estimation_wires
        # eigenvalue = np.exp(2 * np.pi * 1j * theta)

        thetas[i] = theta

    return thetas


def phase_estimation_circuit(unitary: list[list[float]], target_wires: list[int], estimation_wires: list[int],
                             i: int) -> float:
    """
    Phase estimation circuit.

    :param unitary: 2D list of floats:
        The diagonal unitary of which we are trying to estimate the phases.
    :param target_wires: list of ints:
        Our work register, this will hold the answer.
    :param estimation_wires: List of ints:
        Loaded with an eigenstate corresponding to the phase we are after.
    :param i: int:
        We will find the phase corresponding the ith basis vector.

    :return: float:
        A list of lenght len(U) containing the estimated phase.
    """
    # Prepare target wires in the ith basis vector.
    qml.BasisState(np.asarray([i]), wires=target_wires)

    QuantumPhaseEstimation(unitary, target_wires=target_wires, estimation_wires=estimation_wires)

    return qml.probs(estimation_wires)


if __name__ == "__main__":
    from find_cycle_costs import find_cycle_costs

    print("n = 3:")
    A3 = np.asarray([[3, 7, 2],
                     [5, 12, 9],
                     [17, 1, 6]])

    print("\nA:")
    print(A3)

    # Using A, build a unitary matrix where the costs are encoded as phases.
    U_, cycle_cost_range_ = build_U(A=A3, verbose=True)
    print("\nUnitary matrix U:")
    print(U_)
    print("Cycle cost range: " + str(cycle_cost_range_[0]) + ", " + str(cycle_cost_range_[1]))

    # Use phase estimation to obtain the phases (phases encode total costs).
    thetas = phase_estimation(U=U_)

    # Now that we have the phases, convert them back into costs.
    cycle_costs_found = thetas * cycle_cost_range_[1]
    cycle_costs_found += cycle_cost_range_[0]

    print("\nOutput phases:")
    print(cycle_costs_found)

    # print("\nn = 4:")
    # A4 = np.asarray([[3, 7, 2, 19],
    #                  [5, 12, np.nan, 5],
    #                  [17, 1, np.inf, 2.4],
    #                  [3, 13, 13, 12]])
    #
    # print("\nA:")
    # print(A4)
