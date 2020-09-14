#define sqrtlength 20
                    #define const_length sqrtlength *sqrtlength
                    #define off_diagonal_number 5
                    #define array_length const_length *(off_diagonal_number * (-off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)
                    #define big_array_length const_length *(2 * off_diagonal_number * (-2 * off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)