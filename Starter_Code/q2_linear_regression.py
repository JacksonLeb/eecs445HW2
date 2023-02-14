"""
EECS 445 - Introduction to Maching Learning
HW2 Q2 Linear Regression Optimization Methods)
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data
from time import monotonic

def calculate_empirical_risk(X, y, theta):
    loss = 0
    n = X.shape[0]
    for i in range(n):
        loss += (((y[i] - np.dot(theta,  X[i]))**2) / 2)
    return loss / n

def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        theta: np.array, shape (d,). Specifies an (d-1)^th degree polynomial

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    # TODO: Implement this function
    E_rms = 0
    E_err = 0
    n = X.shape[0]
    for i in range(n):
        E_err += (np.dot(theta, X[i]) - y[i])**2
    E_rms = np.sqrt(E_err/n)
    return E_rms


def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector 
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (n, 1). Each row is one instance.
        M: a non-negative integer
    
    Returns:
        Phi: np.array, shape (n, M+1)
    """
    # TODO: Implement this function
    n = X.shape[0]
    Phi = np.zeros((n, M+1))
    if(M == 0):
        return X

    for i, x in enumerate(X):
        Phi[i][0] = 1
        Phi[i][1] = x
        for j in range(2, M + 1):
            Phi[i][j] = pow(x, j)

    return Phi


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
    
    Returns:
        theta: np.array, shape (d,)
    """
    n = X.shape[0]
    d = X.shape[1]
    theta = np.zeros((d,))
    k = 0
    new_loss = 0
    prev_loss = 0
    #convergence criteria
    start_time = monotonic()
    while(k < pow(10, 6)):
        prev_loss = new_loss
        
        y_pred = np.dot(X, theta)
        diff = y_pred - y
        grad = np.dot(X.T, diff) / n


        theta = theta - (learning_rate * grad)
        k += 1

        new_loss = calculate_empirical_risk(X, y, theta)
        
        if np.abs(new_loss - prev_loss) < pow(10, -10): 
            print(f"FINAL TIME: {monotonic() - start_time}")
            print(f"ITERATIONS: {k}")
            return theta
    print(f"FINAL TIME: {monotonic() - start_time}")
    print(f"ITERATIONS: {k}")
    return theta


def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    
    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
    
    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    n = X.shape[0]
    d = X.shape[1]
    theta = np.zeros((d,))
    k = 0
    new_loss = 0
    prev_loss = 0
    #convergence criteria
    start_time = monotonic()
    while(k < pow(10, 6)):
        prev_loss = new_loss
        #np.random.shuffle(X)
        #np.random.shuffle(y)
        for i in range(n):
            theta = theta + ((learning_rate * (y[i] - np.dot(theta, X[i])))*X[i])
            k += 1

        new_loss = calculate_empirical_risk(X, y, theta)
        if np.abs(new_loss - prev_loss) < pow(10, -10): 
            print(f"FINAL TIME: {monotonic() - start_time}")
            print(f"ITERATIONS: {k}")
            return theta
    print(f"FINAL TIME: {monotonic() - start_time}")
    print(f"ITERATIONS: {k}")
    return theta


def closed_form_optimization(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (n, d) 
        y: np.array, shape (n,)
        `reg_param`: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    n = X.shape[0]
    d = X.shape[1]
    theta = np.zeros((d,))

    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + np.identity(d) * reg_param), X.T), y) + (reg_param / 2)
    return theta


def part_2_1(fname_train):
    # TODO: This function should contain all the code you implement to complete 2.1. Please feel free to add more plot commands.
    print("========== Part 2.1 ==========")

    X_train, y_train = load_data(fname_train)
    theta = closed_form_optimization(X=generate_polynomial_features(X_train, 1), y=y_train)
    print("RESULTS: ")
    print(theta)
    poly = np.poly1d(theta)
    y_theta = poly(X_train)
    print("Done!")
    plt.plot(X_train, y_train, 'ro')
    plt.plot(X_train, y_theta, 'b')
    plt.legend()
    plt.savefig('q2_1.png', dpi=200)
    plt.close()


def part_2_2(fname_train, fname_validation):
    # TODO: This function should contain all the code you implement to complete 2.2
    print("=========== Part 2.2 ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    # (a) OVERFITTING

    errors_train = np.zeros((11,))
    errors_validation = np.zeros((11,))
    # Add your code here
    '''
    for i in range(11):
        theta = closed_form_optimization(generate_polynomial_features(X_train, i), y_train)
        errors_train[i] = calculate_RMS_Error(generate_polynomial_features(X_train, i), y_train, theta)
        errors_validation[i] = calculate_RMS_Error(generate_polynomial_features(X_validation, i), y_validation, theta)


    plt.plot(errors_train,'-or',label='Train')
    plt.plot(errors_validation,'-ob', label='Validation')
    plt.xlabel('M')
    plt.ylabel('$E_{RMS}$')
    plt.title('Part 2.2.a')
    plt.legend(loc=1)
    plt.xticks(np.arange(0, 11, 1))
    plt.savefig('q2_2_a.png', dpi=200)
    plt.close()

    '''
    # (b) REGULARIZATION

    errors_train = np.zeros((10,))
    errors_validation = np.zeros((10,))
    L = np.append([0], 10.0 ** np.arange(-8, 1))
    # Add your code here
    best_score = 10000000000000
    best_theta = 0
    for lam in L:
        temp_theta = closed_form_optimization(generate_polynomial_features(X_train, 10), y_train, lam)
        temp_score = calculate_empirical_risk(generate_polynomial_features(X_train, 10), y_train, temp_theta)

        if(temp_score < best_score):
            best_score = temp_score
            best_theta = temp_theta

    for i in range(10):
        theta = closed_form_optimization(generate_polynomial_features(X_train, 10), y_train, L[i])
        errors_train[i] = calculate_RMS_Error(generate_polynomial_features(X_train, 10), y_train, theta)
        errors_validation[i] = calculate_RMS_Error(generate_polynomial_features(X_validation, 10), y_validation, theta)

    plt.figure()
    plt.plot(L, errors_train, '-or', label='Train')
    plt.plot(L, errors_validation, '-ob', label='Validation')
    plt.xscale('symlog', linthresh=1e-8)
    plt.xlabel('$\lambda$')
    plt.ylabel('$E_{RMS}$')
    plt.title('Part 2.2.b')
    plt.legend(loc=2)
    plt.savefig('q2_2_b.png', dpi=200)
    plt.close()

    print("Done!")


def main(fname_train, fname_validation):
    X, y = load_data(fname_train)
    part_2_1(fname_train)
    part_2_2(fname_train, fname_validation)

if __name__ == '__main__':
    main("dataset/q2_train.csv", "dataset/q2_validation.csv")
