sqrtlength_test= 7
const_length_test= sqrtlength_test *sqrtlength_test
off_diagonal_number_test= 2
array_length_test= const_length_test *(off_diagonal_number_test * (-off_diagonal_number_test + 2 * sqrtlength_test - 1) + sqrtlength_test)
big_array_length_test= const_length_test *(2 * off_diagonal_number_test * (-2 * off_diagonal_number_test + 2 * sqrtlength_test - 1) + sqrtlength_test)
