import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    gda = GDA()
    gda.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, gda.theta, 'myOutput/p01e_{}.png'.format(pred_path[-5]))

    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        positive_cnt = y[y == 1.0].shape[0]
        negative_cnt = y[y == 0.0].shape[0]
        sample_cnt = positive_cnt + negative_cnt

        phi = positive_cnt / sample_cnt

        E_negative = np.sum(x[y == 0], axis=0) / negative_cnt
        E_positive = np.sum(x[y == 1], axis=0) / positive_cnt

        covariance = ((x[y == 1] - E_positive).T @ (x[y == 1] - E_positive)) + (
                    (x[y == 0] - E_negative).T @ (x[y == 0] - E_negative))
        covariance /= sample_cnt

        self.theta = np.linalg.inv(covariance) @ (E_positive - E_negative)
        theta_0 = 1 / 2 * (E_positive + E_negative).T @ np.linalg.inv(covariance) @ (E_negative - E_positive) - np.log(
            (1 - phi) / phi)
        self.theta = np.r_[theta_0, self.theta]

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
            # *** END CODE HERE
