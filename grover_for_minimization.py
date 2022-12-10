"""
Gover search algorithm for minimization
"""
import math

import pennylane as qml
from numpy import pi as pi


def minimization_oracle(data_register: list[int], ancillary_register: list[int], arr: list[int], x: int) -> None:
    """
    Marking oracle to flip the signs of the elements that satisfy the minimization condition:
        Mark states |i> (take |i> to -|i>) if arr[i] < x.

    :param data_register: List of ints:
        Our main data qubits.
    :param ancillary_register: List of ints:
        Our helper qubits.
    :param arr: List of ints.
    :param x: int.

    :return: None, operation is done in place.
    """

    def add_k_fourier(k: int, wires: list[int]) -> None:
        """
        Add the integer value k to wires.
        :return: None, operation is done in place.
        """
        for j in range(len(wires)):
            qml.RZ(k * pi / (2 ** j), wires=wires[j])

    def load_values_in_arr():
        """
        Conditionally load all the values in arr onto the axcillary register.
        """
        # Perform a QFT, so we can add values in the Fourier basis.
        qml.QFT(wires=ancillary_register)

        # Loop through each of the data wires, preforming controlled additions.
        for wire in range(len(arr)):
            qml.ctrl(op=add_k_fourier, control=wire)(k=arr[wire], wires=ancillary_register)

        # Return to the computational basis
        qml.adjoint(fn=qml.QFT(wires=ancillary_register))

    load_values_in_arr()

    # Loop thorugh all integers in the range 1 to x-1. Notice we start at 1 because the all 0's state would always be
    #  maked. Anyway, arr should never contain any 0's. If we get a sum of elements on the ancillary wires that is < x,
    #  then there must be at least one element in arr that is less than x.
    for i in range(1, x):
        qml.FlipSign(i, wires=ancillary_register)

    qml.adjoint(fn=load_values_in_arr)()  # Cleanup.


def minimization_circuit(data_register: list[int], ancillary_register: list[int], arr: list[int], x: int) -> list:
    """
    A quantum circuit to find a value in arr that is less than x.

    :param data_register: List of ints:
        Our main data qubits.
    :param ancillary_register: List of ints:
        Our helper qubits.
    :param arr: list of ints.
    :param x: int.

    :return: list:
        A flat array containing the probabilities of measuring each basis state.
    """

    # Step 1: Create an equal superposition by applying a Hadamard to each wire in the data register.
    for wire in data_register:
        qml.Hadamard(wires=wire)

    # for _ in range(4):
    # Step 2: Use the oracle to mark solution states.
    minimization_oracle(data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x)

    # Step 3: Apply the Grover operator to amplify the probability of getting the correct solution.
    qml.GroverOperator(wires=data_register)

    return qml.probs(wires=data_register)


def grover_for_minimization(arr: list[int], x: int) -> bool:
    """
    Use Grover's search to check if arr contains an element < x.

    Important Precondition:
        arr must contain integer values strickly > 0.

    :return: bool:
        True: We found an element in arr < x.
        False: otherwise.
    """
    # We need enough qubits to represent 0 -> sum(arr). You can represent up to 2^n - 1 with n bits
    # This makes sure we can represent all sum compbinations with no wrapping
    number_qubits_required = math.ceil(math.log(sum(arr) + 1, 2))
    # print("number of qubits required:")
    # print(number_qubits_required)

    data_register = list(range(0, number_qubits_required))
    ancillary_register = list(range(number_qubits_required, 2 * number_qubits_required))  # Oracle work space.
    # print(data_register)
    # print(ancillary_register)

    device = qml.device("default.qubit", wires=data_register + ancillary_register)
    qnode = qml.qnode(func=minimization_circuit, device=device)(
        data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x)

    # Go ahead and actually run it.
    values = qnode.__call__(data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x)

    print("\nlen(values):")  # Four bits can encode 16 distinct characters
    print(len(values))
    print("\nvalues:")
    print(values)

    plt.bar(range(len(values)), values)

    plt.show()

    # if result < x:
    #     # If the returned value is actually less than result, great!
    #     return True
    # else:
    #     # No such element exists.
    #     return False

    return False


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    arr_ = [18, 10, 6, 7]
    # arr_ = [3, 5, 4, 3]
    x_ = 63

    found_number_smaller_than_x = grover_for_minimization(arr=arr_, x=x_)

    print("Found a number smaller than x:")
    print(found_number_smaller_than_x)

    # Number of qubits should depend on the sum of arr - make sure we don't roll over!

    # With 4 qubits we can represent 16 numbers (0 -> 15)
    data_register_ = [0, 1, 2, 3]  # 6 qubits
    ancillary_register_ = [4, 5, 6, 7]  # 4 work qubits
