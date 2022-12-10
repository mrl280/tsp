"""
Gover search algorithm for minimization
"""

import pennylane as qml
from numpy import pi as pi


def minimization_oracle(data_register: list[int], ancillary_register: list[int], arr: list[int], x: int) -> None:
    """
    Marking oracle to flip the signs of the elements that satisfy the minimization condition:
        .

    :param data_register: List of ints:
        Our main data qubits.
    :param ancillary_register: List of ints:
        Our helper qubits.
    :param arr: List of ints:
        # TODO
    :param x: int:
        # TODO

    :return: None, operation is done in place.
    """

    def add_k_fourier(k: int, wires: list[int]):
        """
        Add k from wires.

        :param k: int:
            The value to add.

        :param wires:
            The register to add k to.
        """
        for j in range(len(wires)):
            qml.RZ(k * pi / (2 ** j), wires=wires[j])

    def load_values_in_arr():
        """
        ghf.
        """
        # Perform a QFT, so we can add property values in the Fourier basis.
        qml.QFT(wires=ancillary_register)

        # Loop through each of the data registers, preforming controlled additions
        for wire in data_register:
            qml.ctrl(op=add_k_fourier, control=wire)(k=arr[wire], wires=ancillary_register)

        # Return to the computational basis
        qml.adjoint(fn=qml.QFT(wires=ancillary_register))

    load_values_in_arr()
    # If the current value of the ancillary register is a "correct solution", flip the sign

    # If the ancillary_register contains a value smaller than x, we need to flip the sign.
    for i in range(0, x):
        if i in arr:
            qml.FlipSign(i, wires=ancillary_register)
    qml.adjoint(fn=load_values_in_arr)()  # Cleanup.


def minimization_circuit(data_register: list[int], ancillary_register: list[int], arr: list[int], x: int) -> list:
    """
    A quantum circuit to mind a value smaller than x.

    :param data_register: List of ints:
        Our main data qubits.
    :param ancillary_register: List of ints:
        Our helper qubits.
    :param arr: list of ints:
        # TODO
    :param x: int:
        # TODO

    :return: list:
        A flat array containing the probabilities of measuring each basis state.
    """

    # Step 1: Create an equal superposition by applying a Hadamard to each wire in the data register.
    for wire in data_register:
        qml.Hadamard(wires=wire)

    # for _ in range(4):
    # Step 2: Use the oracle to mark elements that < x.
    minimization_oracle(data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x)

    # Step 3: Apply the Grover operator to amplify the probability of getting the correct solution.
    qml.GroverOperator(wires=data_register)

    return qml.probs(wires=data_register)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    arr_ = [12, 10, 6, 7]
    x_ = 10

    # With 4 qubits we can represent 16 numbers (0 -> 15)
    data_register_ = [0, 1, 2, 3]  # 6 qubits
    ancillary_register_ = [4, 5, 6, 7]  # 4 work qubits

    device = qml.device("default.qubit", wires=data_register_ + ancillary_register_)

    qnode = qml.qnode(func=minimization_circuit, device=device)(
        data_register=data_register_, ancillary_register=ancillary_register_, arr=arr_, x=x_)

    values = qnode.__call__(data_register=data_register_, ancillary_register=ancillary_register_, arr=arr_, x=x_)

    print("\nlen(values):")  # Four bits can encode 16 distinct characters
    print(len(values))
    print("\nvalues:")
    print(values)

    plt.bar(range(len(values)), values)

    plt.show()
