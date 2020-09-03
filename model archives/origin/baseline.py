import numpy as np


class Baseline():
    """
    Reward baseline interface
    """

    def get_param_values(self):
        """
        Returns the parameter values of the baseline object

        """
        raise NotImplementedError

    def set_params(self, value):
        """
        Sets the parameter values of the baseline object

        Args:
            value: parameter value to be set

        """
        raise NotImplementedError

    def log_diagnostics(self, paths, prefix):
        """
        Log extra information per iteration based on the collected paths
        """
        pass


class LinearBaseline(Baseline):
    """
    Abstract class providing the functionality for fitting a linear baseline
    Don't instantiate this class. Instead use LinearFeatureBaseline or LinearTimeBaseline
    """

    def __init__(self, reg_coeff=1e-5):
        super(LinearBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def predict(self, obs):
        """
        Abstract Class for the LinearFeatureBaseline and the LinearTimeBaseline
        Predicts the linear reward baselines estimates for a provided trajectory / path.
        If the baseline is not fitted - returns zero baseline

        Args:
           path (dict): dict of lists/numpy array containing trajectory / path information
                 such as "observations", "rewards", ...

        Returns:
             (np.ndarray): numpy array of the same length as paths["observations"] specifying the reward baseline

        """
        if self._coeffs is None:
            return np.zeros(len(obs))
        return self._features(obs).dot(self._coeffs)

    def get_param_values(self, **tags):
        """
        Returns the parameter values of the baseline object

        Returns:
            numpy array of linear_regression coefficients

        """
        return self._coeffs

    def set_params(self, value, **tags):
        """
        Sets the parameter values of the baseline object

        Args:
            value: numpy array of linear_regression coefficients

        """
        self._coeffs = value

    def fit(self, obs, rewards):
        """
        Fits the linear baseline model with the provided paths via damped least squares

        Args:
            paths (list): list of paths
            target_key (str): path dictionary key of the target that shall be fitted (e.g. "returns")

        """
        # assert all([target_key in path.keys() for path in paths])

        featmat = np.concatenate([self._features(ob) for ob in obs], axis=0)
        target = np.concatenate([rw for rw in rewards], axis=0)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(target),
                rcond=-1
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def _features(self, path):
        raise NotImplementedError("this is an abstract class, use either LinearFeatureBaseline or LinearTimeBaseline")


class LinearFeatureBaseline(LinearBaseline):
    """
    Linear (polynomial) time-state dependent return baseline model
    (see. Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control", ICML)

    Fits the following linear model

    reward = b0 + b1*obs + b2*obs^2 + b3*t + b4*t^2+  b5*t^3

    Args:
        reg_coeff: list of paths

    """
    def __init__(self, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def _features(self, obs):
        obs = np.clip(obs, -10, 10)
        path_length = len(obs)
        time_step = np.arange(path_length).reshape(-1, 1) / 100.0
        return np.concatenate([obs, obs ** 2, time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
                              axis=1)


if __name__ == "__main__":

    b = LinearFeatureBaseline()
    o = np.random.randn(3, 2, 2)
    r = np.random.randn(3, 2)
    print(o)
    print(r)
