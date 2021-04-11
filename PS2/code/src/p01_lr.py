# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape
    # margin
    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1. / m) * (X.T.dot(probs * Y))
    # 梯度上升，求最大值

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model"""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 1

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)

        # # learning rate decay
        # learning_rate /= i**2

        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)

            print('Training loss = %f' % np.mean(np.log(1 + np.exp(-Y * X.dot(theta)))))

            print('||theta_k - theta_{k-1}|| = %.15f' % np.linalg.norm(prev_theta - theta))

        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    # # Plot dataset A and B
    from util import plot_points
    import matplotlib.pyplot as plt
    #
    # Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=False)
    # plt.figure()
    # plot_points(Xa, (Ya == 1).astype(int))
    # plt.savefig('output/ds1_a.png')
    #
    # Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)
    # plt.figure()
    # plot_points(Xb, (Yb == 1).astype(int))
    # plt.savefig('output/ds1_b.png')

    # print('==== Training model on data set A ====')
    # Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    # logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    Xb_mean = np.mean(Xb, axis=0)

    Gau_1 = np.random.normal(0, Xb_mean[1] / 10, Xb.shape[0])
    Gau_2 = np.random.normal(0, Xb_mean[2] / 10, Xb.shape[0])
    Xb[:, 1] += Gau_1
    Xb[:, 2] += Gau_2

    plt.figure()
    plot_points(Xb[:, 1:], (Yb == 1).astype(int))
    plt.savefig('output/ds1_b_gaussian.png')

    logistic_regression(Xb, Yb)


if __name__ == '__main__':
    main()
