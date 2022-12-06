"""
https://www.youtube.com/watch?v=KeJqcnpPluc

The inheritance problem, we need to divide a set of properties amongst two siblings such that each inherits an
 equal-value real estate bundle.
"""

import pennylane as qml
from pennylane import numpy as np


def inheritance_oracle(data_register, ancillary_register, property_prices):
    """
    Marking oracle to flip the signs of the elements that satisfy the condition
    :param: data_register:
    :return:
    """

    def add_k_fourier(k, wires):
        for j in range(len(wires)):
            qml.RZ(k * np.pi / (2 ** j), wires=wires[j])

    def value_second_sibling():

        # Perform a QFT so we can add property values in the Fourier basis.
        qml.QFT(wires=ancillary_register)

        # Loop through each of the data registers, preforming controlled additions
        for wire in data_register:
            qml.ctrl(op=add_k_fourier, control=wire)(k=property_prices[wire], wires=ancillary_register)

        # Return to the computational basis
        qml.adjoint(fn=qml.QFT(wires=ancillary_register))

    value_second_sibling()
    # If the current value of the ancillary register is a "correct solution", flip the sign
    qml.FlipSign(sum(property_prices) // 2, wires=ancillary_register)
    qml.adjoint(fn=value_second_sibling)()


property_prices = [4, 8, 6, 3, 12, 15]  # Units in thousands of dollars (totalling 48 thousand)

data_register = [0, 1, 2, 3, 4, 5]  # 6 qubits
ancillary_register = [6, 7, 8, 9, 10, 11]  # 6 more qubits

device = qml.device("default.qubit", wires=data_register + ancillary_register)


@qml.qnode(device)
def circuit():

    # Step 1: Create an equal superposition by applying a Hadamard to each wire in the data register.
    for wire in data_register:
        qml.Hadamard(wires=wire)

    for i in range(4):
        # Step 2: Use the oracle to mark elements that are a correct solution.
        inheritance_oracle(data_register=data_register, ancillary_register=ancillary_register,
                           property_prices=property_prices)

        # Step 3: Apply the Grover operator to amplify the probability of getting the correct solution.
        qml.GroverOperator(wires=data_register)

    return qml.probs(wires=data_register)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    values = circuit()

    print("\nlen(values):")
    print(len(values))
    print("\nvalues:")
    print(values)

    plt.bar(range(len(values)), values)

    plt.show()

