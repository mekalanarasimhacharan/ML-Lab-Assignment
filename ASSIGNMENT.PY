import random
import statistics as stats
import numpy as np
from numpy.linalg import matrix_power


# 1
def find_sum_pairs(data):
    total_pairs = 0
    length = len(data)

    for x in range(length):
        for y in range(x + 1, length):
            if data[x] + data[y] == 10:
                total_pairs += 1
    return total_pairs


# 2
def calculate_range():
    while True:
        try:
            size = int(input("Enter number of elements (must be > 3): "))
            if size <= 3:
                raise ValueError
            break
        except ValueError:
            print("Error: List size should be greater than 3")

    values = []
    for i in range(size):
        val = float(input(f"Enter value {i + 1}: "))
        values.append(val)

    return max(values) - min(values)


# 3
def compute_matrix_power(matrix, exponent):
    while exponent <= 0:
        exponent = int(input("Power must be positive. Enter again: "))
    return matrix_power(matrix, exponent)


# 4
def most_frequent_char(text):
    characters = set(text)
    highest_freq = 0
    frequent_char = None

    for c in characters:
        freq = text.count(c)
        if freq > highest_freq:
            highest_freq = freq
            frequent_char = c

    return frequent_char, highest_freq


# 5
def random_number_stats():
    count = 25
    low, high = 1, 10

    rand_list = [random.randint(low, high) for _ in range(count)]
    rand_list.sort()

    avg = sum(rand_list) / count
    mid = stats.median(rand_list)
    common = stats.mode(rand_list)

    return rand_list, avg, mid, common




# Question 1
numbers = list(map(int, input("Enter numbers: ").split()))
print("Number of pairs with sum 10:", find_sum_pairs(numbers))

# Question 2
print("\nRange of the list:", calculate_range())

# Question 3
mat = np.array([[1, 2], [3, 4]])
p = int(input("\nEnter matrix power: "))
print("Resultant Matrix:\n", compute_matrix_power(mat, p))

# Question 4
user_string = input("\nEnter a string: ")
char, freq = most_frequent_char(user_string)
print(f"Most frequent character: '{char}' occurred {freq} times")

 # Question 5
rand_nums, mean_val, median_val, mode_val = random_number_stats()
print("\nRandom Numbers:", rand_nums)
print("Mean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val)