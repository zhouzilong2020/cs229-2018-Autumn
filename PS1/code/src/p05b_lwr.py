import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***

    lwr = LocallyWeightedLinearRegression(tau=0.5, x=x_train, y=y_train)
    y_pred = lwr.predict(x_eval)

    MSE = np.sum((y_pred - y_eval) ** 2 / y_eval.shape[0])
    print(f"MSE:{MSE}")

    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('myOutput/p05b.png')

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau, x, y):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = x
        self.y = y

    def fit(self, x, y):
        """
        Fit LWR by saving the training set.
        x 是某一个位置
        """

        # *** START CODE HERE ***
        m, n = self.x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        weight = self.getWeight(x)
        inverse = np.linalg.inv(self.x.T @ weight @ self.x)
        self.theta = inverse @ self.x.T @ weight @ self.y.T

        # TODO:使用迭代法确定每次的theta
        # while True:
        #     loss = (y - 1 / (1 + np.exp(-self.x @ self.theta)))
        #     aa = x
        #     a = x * w
        #     theta = self.theta + self.step_size * (self.x * w).T @ loss
        #     diff = np.linalg.norm(self.theta - theta, ord=1)
        #     if diff < self.eps:
        #         break
        #
        #     self.theta = theta

    # *** END CODE HERE ***

    def getWeight(self, x_i):
        """
         x is the point we want to predict at
         self.x is the training dataset
        """
        dividend = 2 * self.tau * self.tau
        dif = self.x - x_i
        weight = np.exp(- np.sum(dif * dif, axis=1) / dividend)
        return np.diag(weight)

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        pre = np.zeros(m)
        for i, pos in enumerate(x):
            self.fit(pos, self.y)
            pre[i] = 1 / (1 + np.exp(-pos @ self.theta))
        return pre
        # *** END CODE HERE ***
