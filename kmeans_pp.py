import math

import numpy as np
import pandas as pd
import sys
import mykmeanssp as km


def merge(file1, file2):
    # read a comma-separated values (csv) file into DataFrame.
    to_merge1 = pd.read_csv(file1, header=None)
    to_merge2 = pd.read_csv(file2, header=None)
    to_merge1.rename(columns={list(to_merge1)[0]: "key"}, inplace=True)
    to_merge2.rename(columns={list(to_merge2)[0]: "key"}, inplace=True)
    result = pd.merge(to_merge1, to_merge2, on="key")
    # Sort by the values along either axis.
    result = result.sort_values("key")
    # Drop specified labels from rows or columns.
    result.drop("key", axis="columns", inplace=True)
    return result.to_numpy()


def buildCentroids(k, n, input_matrix):
    centroids = np.zeros((k, len(input_matrix[0])))
    centroids_index = np.zeros(k)
    # Select µ1 randomly from x1, x2, . . . , xN
    np.random.seed(0)
    random_index = np.random.choice(n, 1)
    # ERROR: setting an array element with a sequence.
    centroids_index[0] = random_index
    centroids[0] = input_matrix[random_index]

    i = 1
    while i < k:
        d = np.zeros(n)
        # Dl = min (xl − µj)^2 ∀j 1 ≤ j ≤ i
        for _ in range(n):
            d[_] = step1(i, input_matrix[_], centroids)
        # randomly select µi = xl, where P(µi = xl) = P(xl)
        prob = np.zeros(n)
        sum_matrix = np.sum(d)
        for j in range(n):
            prob[j] = d[j]/sum_matrix
       #     prob[j] = step2(d[j], sum_matrix)
        rand_i = np.random.choice(n, p=prob)
        centroids_index[i] = rand_i
        centroids[i] = input_matrix[rand_i]
        i += 1

    # The first line will be the indices of the observations chosen by the K-means++ algorithm
    # as the initial centroids. Observation’s index is given by the first column in each input file.
    printIndex(centroids_index)
    return centroids
 

def step1(i, vector, matrix):
    min_vector = math.pow(np.linalg.norm(np.subtract(vector, matrix[0])), 2)
    for j in range(i):
        curr_vector = np.linalg.norm(np.subtract(vector, matrix[j]))
        curr_vector = np.power(curr_vector, 2)
        if min_vector > curr_vector:
            min_vector = curr_vector
    return min_vector


def step2(vector, sum_matrix):
    result = np.divide(vector, sum_matrix)
    return result


def printMatrix(arr):
    for i in range(0, len(arr)):
        for j in range(0, len(arr[0])):
            print(np.round(arr[i][j], 4), end="")
            if j + 1 != len(arr[0]):
                print(",", end="")
        print()


def printIndex(matrix):
    for i in range(0, len(matrix.astype(int))):
        print(matrix.astype(int)[i], end="")
        if i + 1 != len(matrix.astype(int)):
            print(",", end="")
    print()


# k := the number of clusters required.
def execute(k, maxItr, epsilon, input_filename1, input_filename2):
    # combine both input files by inner join using the first column in each file as a key
    input_matrix = merge(input_filename1, input_filename2)
    input_array = input_matrix.tolist()
    # n := number of line of an input file = number of vectors = len(inputMat).
    n = len(input_array)
    # d := number of column of an input file.
    d = len(input_array[0])
    # Check if the data is correct
    if (k >= n) or (maxItr < 0) or (k < 0):
        print("Invalid Input! \n")
    # centroids µ1, µ2, ... , µK ∈ R^d where 1<K<N.
    centroids = buildCentroids(k, n, input_matrix)

    matrix = km.fit(k, maxItr, epsilon, n, d, input_array, centroids.tolist())
    printMatrix(np.array(matrix))


# main
# sys.argv is the list of command-line arguments.
# len(sys.argv) is the number of command-line arguments.
# sys.argv[0] is the program i.e. the script name.
input_argv = sys.argv
input_argc = len(sys.argv)
if input_argc == 6:
    # update max_itr if needed
    # (k, maxItr, epsilon, inputFile1, inputFile2)
    execute(int(input_argv[1]), int(input_argv[2]), float(input_argv[3]), input_argv[4], input_argv[5])
else:
    # (k, epsilon, inputFile1, inputFile2)
    execute(int(input_argv[1]), 300, float(input_argv[2]), input_argv[3], input_argv[4])