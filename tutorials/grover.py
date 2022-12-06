"""
https://www.youtube.com/watch?v=KeJqcnpPluc
"""

import pennylane as qml
from pennylane import numpy as np

property_prices = [4, 8, 6, 3, 12, 15]  # Units in thousands of dollars

variables_wires = [0, 1, 2, 3, 4, 5]  # 6 qubits
aux_oracle_wires = [6, 7, 8, 9, 10, 11]


def oracle():

    def add_k_fourier(k, wires):
        for j in range(len(wires)):
            qml.RZ(k * np.pi / (2 ** j), wires=wires[j])

        def value_second_sibling():

            qml.QFT(wires=aux_oracle_wires)

            for wire in variables_wires:
                qml.ctrl(add_k_fourier, control=wire)(
                    property_prices[wire],
                    wires=aux_oracle_wires
                )

            qml.adjoint(qml.QFT)(wires=aux_oracle_wires)

        value_second_sibling()
        qml.FlipSign(sum(property_prices) // 2, wires=aux_oracle_wires)
        qml.adjoint(value_second_sibling())


if __name__ == "__main__":
    print(np.pi)

