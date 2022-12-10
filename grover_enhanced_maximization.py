"""
Find the minimum value in a list using the Durr & Hoyer Quantum algorithm for minimzation

C. Durr and P. Hoyer, “A Quantum Algorithm for Finding the Minimum,” 1996, doi: 10.48550/arxiv.quant-ph/9607014.

Binary search code courtesy of https://www.geeksforgeeks.org/python-program-for-binary-search/.
"""


def grover_enhanced_minimization(arr: list[int], lower_bound: int = 0, upper_bound=None) -> int:
    """
    Use the Durr & Hoyer Quantum algorithm for minimzation to find the minimum value in arr.

    This algorithm is basically just binary search ontop of Grover's search.

    # TODO: Expand functionality to encompase all comparable types.

    :param arr: list of ints:
        A list of comparable input values. We are looking for the minimum.

    :param lower_bound: int or float (optional; default is 0):


    :return: int:
        The index of the smallest value in arr.
    """
    if upper_bound is None:  # First iteration
        upper_bound = arr[0]

    print("\nLower bound: " + str(lower_bound))
    print("Upper bound: " + str(upper_bound))

    # Check base case
    if upper_bound >= lower_bound:

        middle = (upper_bound + lower_bound) // 2

        # Check if there is an element smaller than the middle value.
        smaller_element_exists = classical_grover(arr=arr, x=middle)
        if smaller_element_exists:
            # There is an element smaller than middle, lower our upper bound.
            print("Smaller element found, lowering our upper bound...")
            return grover_enhanced_minimization(arr, lower_bound=lower_bound, upper_bound=middle - 1)

        else:
            # There are no elements smaller than middle, raise the lower bound
            print("No smaller element found, raising our lower bound...")
            return grover_enhanced_minimization(arr, lower_bound=middle + 1, upper_bound=upper_bound)

    else:
        # Check if middle exists in array
        # return classical_grover(arr=arr, x=middle)
        return upper_bound


def classical_grover(arr, x) -> bool:
    """
    For testing, this is a classical function to perform the same functionality as Grover will.

    We are looking for a value in arr smaller than x.

    :return:
        True: if we find an element in arr < x
        False: otherwise.
    """
    for i in range(len(arr)):
        if arr[i] < x:
            return True  # We have found a smaller value

    return False  # No element in list > middle


if __name__ == "__main__":

    test_array = [6, 15, 17, 8, 11, 2, 9]
    result = grover_enhanced_minimization(arr=test_array, lower_bound=0, upper_bound=None)

    print(result)
