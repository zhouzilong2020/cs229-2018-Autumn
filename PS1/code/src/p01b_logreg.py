import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    logistic = LogisticRegression(eps=1e-5)
    logistic.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, logistic.theta, 'myOutput/p01b_{}.png'.format(pred_path[-5]))

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = logistic.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def gradient(self, x, y, h_x):
        return x.T @ (h_x - y)

    def hessain(self, x, h_x):
        # a = x.T * h_x
        # b = a * (1 - h_x)
        return x.T * h_x * (1 - h_x) @ x

    def prompt(self, x, y, predict):
        predict = np.around(predict)
        cnt = y[predict == y].shape[0]
        print(f"precision:{cnt / (y.shape[0])}")

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # init theta
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        epsilon = 1

        while epsilon > self.eps:
            h_x = self.predict(x)
            if self.verbose:
                self.prompt(x, y, h_x)
            H = self.hessain(x, h_x)
            g = self.gradient(x, y, h_x)
            theta = self.theta - np.linalg.inv(H) @ g
            epsilon = np.linalg.norm(theta - self.theta, ord=1)
            self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x @ self.theta))
        # *** END CODE HERE ***
