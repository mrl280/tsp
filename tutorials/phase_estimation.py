"""
Practice with the PennyLane documentation's phase estimation problem:
https://docs.pennylane.ai/en/stable/code/api/pennylane.QuantumPhaseEstimation.html
"""

import pennylane as qml
from pennylane.templates import QuantumPhaseEstimation
from pennylane import numpy as np

phase = 5  # The error in our estimate will decrease exponentially with the number of qubits.
target_wires = [0]
unitary = qml.RX(phase, wires=0).matrix()

print("Here is the unitary matrix of which we are trying to estimate the phase:")
print(unitary)

n_estimation_wires = 5
estimation_wires = range(1, n_estimation_wires + 1)

dev = qml.device("default.qubit", wires=n_estimation_wires + 1)


@qml.qnode(dev)
def circuit():
    """
    A quantum circuit to estimate the phase of an Rx gate.
    :return: Probability distribution over measurement outcomes in the computational basis.
    """
    # Start in the |+> eigenstate of the unitary
    qml.Hadamard(wires=target_wires)

    QuantumPhaseEstimation(
        unitary,
        target_wires=target_wires,
        estimation_wires=estimation_wires,
    )

    return qml.probs(estimation_wires)


# Find the index of the largest value in the probability distribution and divide that number by 2^n.
phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

# Need to rescale phase due to some convention of RX gate.
phase_estimated = 4 * np.pi * (1 - phase_estimated)
print("Phase estimated: " + str(phase_estimated))

# From the phase, we can recove the eigenvalue!
matrix_eigenvalue = np.exp(2 * np.pi * 1j * theta)
print("Matrix eigenvalue: " + str(matrix_eigenvalue))
