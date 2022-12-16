"""
Author: Michael Luciuk
Date:   Dec, 2022

Gover search for minimization. Uses the Long et al. improvement to increase probability of success for a large number
 of cities.

L. Grover, “A fast quantum mechanical algorithm for database search,” in Proceedings of the twenty-eighth annual ACM
 symposium on theory of computing, 1996, pp. 212–219. doi: 10.1145/237814.237866.

G. L. Long, “Grover algorithm with zero theoretical failure rate,” Physical review. A, Atomic, molecular, and optical
 physics, vol. 64, no. 2, p. 022307/4–, 2001, doi: 10.1103/PhysRevA.64.022307.
"""
import math

import pennylane as qml
import matplotlib.pyplot as plt

from numpy import pi as pi


def minimization_oracle(data_register: list[int], ancillary_register: list[int], arr: list[int], x: int,
                        phi: float) -> None:
    """
    Originally, this marking oracle would flip the signs of the elements that satisfies the minimization condition.
        That is: mark states |i> (take |i> to -|i>) if arr[i] < x.

    With the Long et al. upgrade, this phase inversion is now replaces by a phase rotation through angle phi.

    :param data_register: List of ints:
        Our main data qubits.
    :param ancillary_register: List of ints:
        Our helper qubits.
    :param arr: List of ints.
    :param x: int.
    :param phi: flaot:
        As explained in Long et al., we can replace the classic phase inversion by a phase rotation through angle phi
         to increase our success rate.

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
        for wire in data_register:
            qml.ctrl(op=add_k_fourier, control=wire)(k=arr[wire], wires=ancillary_register)

        # Return to the computational basis.
        qml.adjoint(fn=qml.QFT(wires=ancillary_register))

    load_values_in_arr()

    # Loop thorugh all integers in the range 1 to x-1. Notice we start at 1 because the all 0's state would always be
    #  marked. Anyway, in the TSP context arr should never contain any 0's. If we get a sum of elements on the
    #  ancillary wires that is < x, then there must be at least one element in arr that is less than x.
    for i in range(1, x):

        # qml.FlipSign(i, wires=ancillary_register)
        # To improve sucess, we replace the phase inversion by a phase rotation through phi.

        # Implement flip_sign in pauli primatives
        arr_bin = to_list(n=i, n_wires=len(ancillary_register))  # Turn i into a state.

        if arr_bin[-1] == 0:
            qml.PauliX(wires=ancillary_register[-1])

        # qml.ctrl(qml.PauliZ, control=ancillary_register[:-1], control_values=arr_bin[:-1])(
        #             wires=ancillary_register[-1])

        # Perform with RZ instead of Pauli-Z so we can do an arbitrary angle.
        if arr_bin[-1] == 0:
            qml.RZ(phi=phi, wires=ancillary_register[-1])
        qml.ctrl(qml.PauliX, control=ancillary_register[:-1], control_values=arr_bin[:-1])(
            wires=ancillary_register[-1])
        if arr_bin[-1] == 0:
            qml.RZ(phi=-phi, wires=ancillary_register[-1])
        qml.ctrl(qml.PauliX, control=ancillary_register[:-1], control_values=arr_bin[:-1])(
            wires=ancillary_register[-1])

        if arr_bin[-1] == 0:
            qml.PauliX(wires=ancillary_register[-1])

    qml.adjoint(fn=load_values_in_arr)()  # Cleanup.


def to_list(n, n_wires):
    """
    This function is copy and pasted straight out of PennyLane!

    Convert an integer into a binary integer list
    Args:
        n (int): Basis state as integer
        n_wires (int): Numer of wires to transform the basis state
    Raises:
        ValueError: "cannot encode n with n wires "
    Returns:
        (array[int]): integer binary array
    """
    # From PennyLane
    if n >= 2 ** n_wires:
        raise ValueError(f"cannot encode {n} with {n_wires} wires ")

    b_str = f"{n:b}".zfill(n_wires)
    bin_list = [int(i) for i in b_str]
    return bin_list


def minimization_circuit(data_register: list[int], ancillary_register: list[int], arr: list[int], x: int,
                         ret: str = "sample") -> list:
    """
    A quantum circuit to find a value in arr that is less than x.

    :param data_register: List of ints:
        Our main data qubits.
    :param ancillary_register: List of ints:
        Our helper qubits.
    :param arr: list of ints.
    :param x: int.

    :param ret: str (optional; default is "sample"):
        "sample": return sample.
        "probs": return probabilities.

    :return: Depends on input paramter ret.
    """

    # Step 1: Create an equal superposition by applying a Hadamard to each wire in the data register.
    for wire in data_register:
        qml.Hadamard(wires=wire)

    # Compute some of the parameters defined in Long et al.
    beta = math.asin(1 / math.sqrt(len(arr)))
    j_op = math.floor((pi / 2 - beta) / (2 * beta))
    j = j_op
    phi = 2 * math.asin(math.sin(pi / (4 * j + 6)) / math.sin(beta))

    print("phi: " + str(phi))
    print("j_op: " + str(j_op))
    print("j: " + str(j))
    print("beta: " + str(beta))
    # Upon measurment at the (J + 1)th iteration, the marked state is obtained with certainty.

    for _ in range(j+1):
        # Step 2: Use the oracle to mark solution states.
        minimization_oracle(data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x, phi=phi)

        # Step 3: Apply the Grover operator to amplify the probability of getting the correct solution.
        qml.GroverOperator(wires=data_register)

    if ret == "probs":
        return qml.probs(wires=data_register)
    else:
        return qml.sample(wires=data_register)


def grover_for_minimization(arr: list[int], x: int) -> bool:
    """
    Use Grover's search to check if arr contains an element < x.

    Important Precondition:
        arr must contain integer values strickly > 0.

    :return: bool:
        True: We found an element in arr < x.
        False: otherwise.
    """
    # We need one wire in the data register for each element of arr.
    number_of_data_qubits_required = len(arr)

    # We need enough ancillary qubits to represent 0 -> sum(arr) or x, whichever is larger.
    #  Recall you can represent up to 2^n - 1 with n bits.
    number_ancillary_qubits_required = max(math.ceil(math.log(sum(arr) + 1, 2)), math.ceil(math.log(x + 1, 2)))
    # print("number of qubits required:")
    # print(number_qubits_required)

    data_register = list(range(0, number_of_data_qubits_required))  # One data qubit for each element of arr
    ancillary_register = list(range(number_of_data_qubits_required,
                                    number_of_data_qubits_required + number_ancillary_qubits_required))
    # print(data_register)
    # print(ancillary_register)

    # We will create 2 devices, one first for creating a plot of the probabilities.
    device1 = qml.device("default.qubit", wires=data_register + ancillary_register)
    qnode1 = qml.qnode(func=minimization_circuit, device=device1)(
        data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x, ret="probs")

    # Obtain and plot a probablity distribution.
    values = qnode1.__call__(data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x,
                             ret="probs")
    print("\nlen(values):")  # Four bits can encode 16 distinct characters
    print(len(values))
    print("\nvalues:")
    print(values)
    plt.bar(range(len(values)), values)
    plt.show()

    # The second device for actully obtaining a sample.
    device2 = qml.device("default.qubit", wires=data_register + ancillary_register, shots=1)
    qnode2 = qml.qnode(func=minimization_circuit, device=device2)(
        data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x, ret="sample")

    sample = qnode2.__call__(data_register=data_register, ancillary_register=ancillary_register, arr=arr, x=x,
                             ret="sample")
    print(sample)

    found_elements = [i for indx, i in enumerate(arr) if sample[indx] == 1]
    print("Found elements: " + str(found_elements))

    if len(found_elements) > 0:
        if found_elements[0] < x:
            # If the found value is actually less than result, great!
            return True

    # No such element exists.
    return False


if __name__ == "__main__":

    # arr_ = [18, 10, 6, 7]
    # arr_ = [3, 5, 4, 3]
    arr_ = [36, 5, 8, 14, 15, 2, 4, 16]
    x_ = 3

    found_number_smaller_than_x = grover_for_minimization(arr=arr_, x=x_)

    print("Found a number smaller than x: " + str(found_number_smaller_than_x))
