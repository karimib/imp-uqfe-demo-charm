import unittest
from qfehelpers import modular_inverse, matrix_vector_dot, inner_product_mod, transpose_matrix, random_int_matrix, random_vector, matrix_vector_multiply

class TestModularInverse(unittest.TestCase):
    def test_valid_cases(self):
        # Test cases with valid modular inverses
        self.assertEqual(modular_inverse(3, 7), 5)  # 3 * 5 % 7 == 1
        self.assertEqual(modular_inverse(2, 13), 7)  # 2 * 7 % 13 == 1
        self.assertEqual(modular_inverse(10, 17), 12)  # 10 * 12 % 17 == 1
        self.assertEqual(modular_inverse(1, 19), 1)  # 1 * 1 % 19 == 1

    def test_no_inverse(self):
        # Test cases where no modular inverse exists
        with self.assertRaises(ValueError):
            modular_inverse(6, 9)  # 6 and 9 are not coprime

        with self.assertRaises(ValueError):
            modular_inverse(4, 8)  # 4 and 8 are not coprime

    def test_edge_cases(self):
        # Edge cases
        self.assertEqual(modular_inverse(1, 5), 1)  # 1 * 1 % 5 == 1
        self.assertEqual(modular_inverse(4, 5), 4)  # 4 * 4 % 5 == 1

    def test_large_numbers(self):
        # Test cases with large numbers
        p = 16283262548997601220198008118239886026907663399064043451383740756301306087801
        a = 123456789123456789
        inverse = modular_inverse(a, p)
        self.assertEqual((a * inverse) % p, 1)  # Check if the modular inverse is valid


class TestMatrixVectorDot(unittest.TestCase):
    def test_valid_cases(self):
        # Test cases with valid matrix and vector sizes
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        vector = [1, 2, 3]
        p = 10
        expected_result = [4, 2, 0]  # Computed mod 10
        self.assertEqual(matrix_vector_dot(matrix, vector, p), expected_result)

        matrix = [
            [2, 0],
            [1, 3],
            [4, 5]
        ]
        vector = [3, 4]
        p = 7
        expected_result = [6, 1, 4]  # Computed mod 7
        self.assertEqual(matrix_vector_dot(matrix, vector, p), expected_result)

    def test_mismatched_dimensions(self):
        # Test case where matrix and vector dimensions don't match
        matrix = [
            [1, 2],
            [3, 4]
        ]
        vector = [1, 2, 3]
        p = 7
        with self.assertRaises(ValueError):
            matrix_vector_dot(matrix, vector, p)

    def test_empty_matrix_or_vector(self):
        # Test cases with empty matrix or vector
        matrix = []
        vector = [1, 2, 3]
        p = 7
        with self.assertRaises(IndexError):  # Accessing len(matrix[0]) will fail
            matrix_vector_dot(matrix, vector, p)

        matrix = [[1, 2], [3, 4]]
        vector = []
        with self.assertRaises(ValueError):  # Mismatched dimensions
            matrix_vector_dot(matrix, vector, p)

    def test_single_row_matrix(self):
        # Test case with a single-row matrix
        matrix = [[1, 2, 3]]
        vector = [4, 5, 6]
        p = 100
        expected_result = [32]  # Computed mod 100
        self.assertEqual(matrix_vector_dot(matrix, vector, p), expected_result)

    def test_large_numbers(self):
        # Test case with large numbers
        matrix = [
            [123456789, 987654321],
            [111111111, 222222222]
        ]
        vector = [333333333, 444444444]
        p = 10**9 + 7
        expected_result = [159122116, 913580262]  # Computed mod 10**9+7
        self.assertEqual(matrix_vector_dot(matrix, vector, p), expected_result)


class TestInnerProductMod(unittest.TestCase):
    def test_valid_cases(self):
        # Test valid inner product modulo p
        vector1 = [1, 2, 3]
        vector2 = [4, 5, 6]
        p = 7
        expected_result = (1*4 + 2*5 + 3*6) % 7  # (4 + 10 + 18) % 7 = 32 % 7 = 4
        self.assertEqual(inner_product_mod(vector1, vector2, p), expected_result)

        vector1 = [2, 3, 4]
        vector2 = [5, 6, 7]
        p = 13
        expected_result = (2*5 + 3*6 + 4*7) % 13  # (10 + 18 + 28) % 13 = 56 % 13 = 4
        self.assertEqual(inner_product_mod(vector1, vector2, p), expected_result)

    def test_mismatched_lengths(self):
        # Test mismatched vector lengths
        vector1 = [1, 2]
        vector2 = [3, 4, 5]
        p = 7
        with self.assertRaises(ValueError):
            inner_product_mod(vector1, vector2, p)

    def test_zero_modulus(self):
        # Test with modulus 1 (everything mod 1 is 0)
        vector1 = [1, 2, 3]
        vector2 = [4, 5, 6]
        p = 1
        self.assertEqual(inner_product_mod(vector1, vector2, p), 0)

    def test_empty_vectors(self):
        # Test with empty vectors
        vector1 = []
        vector2 = []
        p = 7
        self.assertEqual(inner_product_mod(vector1, vector2, p), 0)

    def test_large_numbers(self):
        # Test with large numbers
        vector1 = [10**9 + 7, 10**9 + 9]
        vector2 = [10**9 + 6, 10**9 + 5]
        p = 10**9 + 7
        expected_result = ((10**9 + 7) * (10**9 + 6) + (10**9 + 9) * (10**9 + 5)) % (10**9 + 7)
        self.assertEqual(inner_product_mod(vector1, vector2, p), expected_result)

    def test_edge_cases(self):
        # Test vectors with one element
        vector1 = [1]
        vector2 = [1]
        p = 5
        self.assertEqual(inner_product_mod(vector1, vector2, p), 1)

        vector1 = [0]
        vector2 = [0]
        p = 5
        self.assertEqual(inner_product_mod(vector1, vector2, p), 0)


class TestTransposeMatrix(unittest.TestCase):
    def test_valid_square_matrix(self):
        # Test case for a square matrix
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        expected_result = [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ]
        self.assertEqual(transpose_matrix(matrix), expected_result)

    def test_valid_rectangular_matrix(self):
        # Test case for a rectangular matrix (more rows than columns)
        matrix = [
            [1, 2],
            [3, 4],
            [5, 6]
        ]
        expected_result = [
            [1, 3, 5],
            [2, 4, 6]
        ]
        self.assertEqual(transpose_matrix(matrix), expected_result)

        # Test case for a rectangular matrix (more columns than rows)
        matrix = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        expected_result = [
            [1, 4],
            [2, 5],
            [3, 6]
        ]
        self.assertEqual(transpose_matrix(matrix), expected_result)

    def test_single_row_matrix(self):
        # Test case for a single row matrix
        matrix = [[1, 2, 3]]
        expected_result = [
            [1],
            [2],
            [3]
        ]
        self.assertEqual(transpose_matrix(matrix), expected_result)

    def test_single_column_matrix(self):
        # Test case for a single column matrix
        matrix = [
            [1],
            [2],
            [3]
        ]
        expected_result = [[1, 2, 3]]
        self.assertEqual(transpose_matrix(matrix), expected_result)

    def test_empty_matrix(self):
        # Test case for an empty matrix
        matrix = []
        with self.assertRaises(ValueError):
            transpose_matrix(matrix)

    def test_matrix_with_empty_rows(self):
        # Test case for a matrix with empty rows
        matrix = [[]]
        with self.assertRaises(ValueError):
            transpose_matrix(matrix)

    def test_large_matrix(self):
        # Test case for a large matrix
        matrix = [[i + j for j in range(100)] for i in range(100)]
        expected_result = [[matrix[j][i] for j in range(100)] for i in range(100)]
        self.assertEqual(transpose_matrix(matrix), expected_result)



class TestRandomIntMatrix(unittest.TestCase):
    def test_valid_dimensions(self):
        # Test valid dimensions with low and high range
        low, high, n, k = 1, 10, 3, 4
        matrix = random_int_matrix(low, high, n, k)
        
        # Check dimensions
        self.assertEqual(len(matrix), n)
        self.assertTrue(all(len(row) == k for row in matrix))

        # Check if all elements are within range
        for row in matrix:
            for value in row:
                self.assertGreaterEqual(value, low)
                self.assertLess(value, high)

    def test_single_row(self):
        # Test case for a single-row matrix
        low, high, n, k = 5, 15, 1, 4
        matrix = random_int_matrix(low, high, n, k)
        
        # Check dimensions
        self.assertEqual(len(matrix), n)
        self.assertEqual(len(matrix[0]), k)

        # Check if all elements are within range
        for value in matrix[0]:
            self.assertGreaterEqual(value, low)
            self.assertLess(value, high)

    def test_single_column(self):
        # Test case for a single-column matrix
        low, high, n, k = 10, 20, 5, 1
        matrix = random_int_matrix(low, high, n, k)
        
        # Check dimensions
        self.assertEqual(len(matrix), n)
        self.assertTrue(all(len(row) == k for row in matrix))

        # Check if all elements are within range
        for row in matrix:
            self.assertGreaterEqual(row[0], low)
            self.assertLess(row[0], high)

    def test_empty_matrix(self):
        # Test case for an empty matrix
        low, high, n, k = 1, 10, 0, 0
        matrix = random_int_matrix(low, high, n, k)
        self.assertEqual(matrix, [])

    def test_large_matrix(self):
        # Test case for a large matrix
        low, high, n, k = 0, 100, 100, 100
        matrix = random_int_matrix(low, high, n, k)
        
        # Check dimensions
        self.assertEqual(len(matrix), n)
        self.assertTrue(all(len(row) == k for row in matrix))

    def test_invalid_range(self):
        # Test case for invalid range (low >= high)
        low, high, n, k = 10, 5, 3, 4
        with self.assertRaises(ValueError):
            random_int_matrix(low, high, n, k)


class TestRandomVector(unittest.TestCase):
    def test_valid_vector(self):
        # Test case for a valid vector
        low, high, n = 1, 10, 5
        vector = random_vector(low, high, n)

        # Check length
        self.assertEqual(len(vector), n)

        # Check if all elements are within range
        for value in vector:
            self.assertGreaterEqual(value, low)
            self.assertLess(value, high)

    def test_single_element_vector(self):
        # Test case for a single-element vector
        low, high, n = 5, 10, 1
        vector = random_vector(low, high, n)

        # Check length
        self.assertEqual(len(vector), n)

        # Check if the element is within range
        self.assertGreaterEqual(vector[0], low)
        self.assertLess(vector[0], high)

    def test_empty_vector(self):
        # Test case for an empty vector
        low, high, n = 1, 10, 0
        vector = random_vector(low, high, n)

        # Check that the vector is empty
        self.assertEqual(vector, [])

    def test_large_vector(self):
        # Test case for a large vector
        low, high, n = 1, 100, 1000
        vector = random_vector(low, high, n)

        # Check length
        self.assertEqual(len(vector), n)

        # Check if all elements are within range
        for value in vector:
            self.assertGreaterEqual(value, low)
            self.assertLess(value, high)

    def test_invalid_range(self):
        # Test case for invalid range (low >= high)
        low, high, n = 10, 5, 5
        with self.assertRaises(ValueError):
            random_vector(low, high, n)

    def test_zero_range(self):
        # Test case for zero range (low == high)
        low, high, n = 7, 7, 5
        with self.assertRaises(ValueError):
            random_vector(low, high, n)


class TestMatrixVectorMultiply(unittest.TestCase):
    def test_basic_case(self):
        matrix = [[1, 2], [3, 4]]
        vector = [5, 6]
        expected = [17, 39]
        self.assertEqual(matrix_vector_multiply(matrix, vector), expected)

    def test_larger_matrix(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]
        vector = [1, 2, 3]
        expected = [14, 32, 50, 68]
        self.assertEqual(matrix_vector_multiply(matrix, vector), expected)

    def test_single_element_case(self):
        matrix = [[10]]
        vector = [2]
        expected = [20]
        self.assertEqual(matrix_vector_multiply(matrix, vector), expected)

    def test_mismatched_dimensions(self):
        matrix = [[1, 2, 3], [4, 5, 6]]
        vector = [1, 2]
        with self.assertRaises(ValueError) as context:
            matrix_vector_multiply(matrix, vector)
        self.assertEqual(
            str(context.exception),
            "Number of columns in the matrix must match the length of the vector"
        )

    def test_zeros(self):
        matrix = [[0, 0, 0], [0, 0, 0]]
        vector = [0, 0, 0]
        expected = [0, 0]
        self.assertEqual(matrix_vector_multiply(matrix, vector), expected)

    def test_empty_inputs(self):
        matrix = []
        vector = []
        with self.assertRaises(IndexError):
            matrix_vector_multiply(matrix, vector)


if __name__ == "__main__":
    unittest.main()
