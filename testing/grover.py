"""
https://www.youtube.com/watch?v=KeJqcnpPluc

The inheritance problem: we need to divide a set of properties amongst two siblings such that each inherits an
 equal-value real estate bundle.
"""
import math

import pennylane as qml
from numpy import pi as pi


def inheritance_oracle(data_register: list[int], ancillary_register: list[int],
                       property_prices: list[int | float]) -> None:
    """
    Marking oracle to flip the signs of the elements that satisfy the inheritence condition:
        The value of the second silbings relestate bundle is equal to half the total inhertance.

    :param data_register: List of ints:
        Our main data qubits.
    :param ancillary_register: List of ints:
        Our helper qubits.
    :param property_prices: List of ints or floats:
        A list of prices of the properties to be split among the sibilings.

    :return: None, operation is done in place.
    """

    def add_k_fourier(k: int, wires: list[int]):
        """
        Add k to wires.

        :param k: int:
            The value to add.

        :param wires:
            The register to add k to.
        """
        for j in range(len(wires)):
            qml.RZ(k * pi / (2 ** j), wires=wires[j])

    def value_second_sibling():
        """
        Evaluate the value of the second siblings property holding using a series of controlled rotations.
        """
        # Perform a QFT, so we can add property values in the Fourier basis.
        qml.QFT(wires=ancillary_register)

        # Loop through each of the data registers, preforming controlled additions.
        for wire in data_register:
            qml.ctrl(op=add_k_fourier, control=wire)(k=property_prices[wire], wires=ancillary_register)

        # Return to the computational basis.
        qml.adjoint(fn=qml.QFT(wires=ancillary_register))

    value_second_sibling()
    # If the current value of the ancillary register is a "correct solution", flip the sign.
    qml.FlipSign(sum(property_prices) // 2, wires=ancillary_register)
    qml.adjoint(fn=value_second_sibling)()  # Cleanup.


def inheritence_circuit(data_register: list[int], ancillary_register: list[int],
                        property_prices: list[int | float]) -> list:
    """
    A quantum circuit to solve the inheritence problem.

    :param data_register: List of ints:
        Our main data qubits.
    :param ancillary_register: List of ints:
        Our helper qubits.
    :param property_prices: List of ints or floats:
        A list of prices of the properties to be split among the sibilings.

    :return: list:
        A flat array containing the probabilities of measuring each basis state. Since we have 6 qubits, we expect a
         list of 64 probablities.
    """

    n = len(data_register)
    N = 2 ** n
    M = 2  # 2 marked elements (the same solution, just with the respective real estate bundles swapped)

    # Step 1: Create an equal superposition by applying a Hadamard to each wire in the data register.
    for wire in data_register:
        qml.Hadamard(wires=wire)

    # Repeat the oracle-operator pairing an optimal number of times.
    optimal_number_of_grover_iterations = math.ceil(pi / 4 * math.sqrt(N / M) - 1 / 2)
    print("Optimal number of Grover iterations: " + str(optimal_number_of_grover_iterations))
    for _ in range(optimal_number_of_grover_iterations):
        # Step 2: Use the oracle to mark elements that are a correct solution.
        inheritance_oracle(data_register=data_register, ancillary_register=ancillary_register,
                           property_prices=property_prices)

        # Step 3: Apply the Grover operator to amplify the probability of getting the correct solution.
        qml.GroverOperator(wires=data_register)

    return qml.probs(wires=data_register)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    property_prices_ = [4, 8, 6, 3, 12, 15]  # Units in thousands of dollars (totalling 48 thousand)

    data_register_ = [0, 1, 2, 3, 4, 5]  # 6 data qubits
    ancillary_register_ = [6, 7, 8, 9, 10, 11]  # 6 work qubits

    device = qml.device("default.qubit", wires=data_register_ + ancillary_register_)

    qnode = qml.qnode(func=inheritence_circuit, device=device)(data_register=data_register_,
                                                               ancillary_register=ancillary_register_,
                                                               property_prices=property_prices_)

    values = qnode.__call__(data_register=data_register_, ancillary_register=ancillary_register_,
                            property_prices=property_prices_)

    print("\nlen(values):")  # Six bits can encode 64 distinct characters
    print(len(values))
    print("\nvalues:")
    print(values)

    # Plot the probability distribution.
    plt.bar(range(len(values)), values)
    plt.show()
