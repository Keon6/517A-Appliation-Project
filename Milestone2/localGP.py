from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     ShuffleSplit, StratifiedShuffleSplit)
from sklearn.gaussian_process.kernels import (Matern, RationalQuadratic,
                                              ExpSineSquared)
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree
import numpy as np


class LocalGaussianProcessClassifier:
    def __init__(self, kernel_hyperparams, kernel_type="Matern", k=100):
        """
        :param kernel_hyperparams: list of hyperparameters to input
        :param kernel_type: str for kernel type
        :param k: value for k-NN
        """
        self.kernel_hyperparams = kernel_hyperparams
        self.kernel_type = kernel_type
        self.k = k
        self.kernel_map = {
            "Matern": Matern(
                kernel_hyperparams[0], kernel_hyperparams[1],
                kernel_hyperparams[2]
            ),
            "RationalQuadratic": RationalQuadratic(
                kernel_hyperparams[0], kernel_hyperparams[1],
                kernel_hyperparams[2], kernel_hyperparams[3]
            ),
            "ExpSineSquared": ExpSineSquared(
                kernel_hyperparams[0], kernel_hyperparams[1],
                kernel_hyperparams[2], kernel_hyperparams[3]
            )
        }

    def predict(self, X_tr, X_te, Y_tr):
        """
        There is no "Model fitting" phase for local GP
        :param X_tr: Training Data (n x d)
        :param X_te: Test Data (m x d)
        :param Y_tr: Training Labels (n x 1)
        :return: prediction labels (n x 1)
        """
        # runn k-NN first
        tree = BallTree(X_tr, leaf_size=2)
        dist, ind = tree.query(X_te, k=self.k)
        # GP Classification
        model = GaussianProcessClassifier(
            kernel=self.kernel_map[self.kernel_type],
            random_state=0, n_jobs=-1
        )
        Y_prediction = []
        for index_list, x_new in zip(ind, np.asarray(X_te)):
            x_new = x_new.reshape(1, -1)
            X = np.asarray(X_tr)[index_list, :]
            Y = np.asarray(Y_tr)[index_list]
            if sum(Y) == 0:
                Y_prediction.append(0)
            elif sum(Y) == 100:
                Y_prediction.append(1)
            else:
                model = model.fit(X, Y)
                Y_prediction.append(model.predict(x_new)[0])
        return np.asarray(Y_prediction).reshape(-1, 1)

    def score(self, X_tr, Y_tr, score_criteria):
        """
        score for training data
        :param X_tr:
        :param Y_tr:
        :param score_criteria: criteria for scoring ie) roc_auc_score, ...
        :return: Returns score for training data (float)
        """
        Y_pred = self.predict(X_tr, X_tr, Y_tr)
        return score_criteria(Y_tr, Y_pred)

    def test_score(self, X_tr, Y_tr, X_te, Y_te, score_criteria):
        """
        :param X_tr:
        :param Y_tr:
        :param X_te:
        :param Y_te:
        :param score_criteria: criteria for scoring ie) roc_auc_score, ...
        :return: returns test score
        """
        Y_pred = self.predict(X_tr, X_te, Y_tr)
        return score_criteria(Y_te, Y_pred)


class LGPC_CV:
    """ CV for Local Gaussian Process Calssifier
    """

    def __init__(self, model, score_criteria=roc_auc_score, n_splits=5, split_method=StratifiedKFold):
        """
        :param model: model object for LGPC
        :param n_splits: number of splits for CV
        :param score_criteria: object for scoring criteria ie) roc_auc_score
        :param split_method:
        """
        self.n_splits = n_splits
        self.model = model
        self.score_criteria = score_criteria
        self.split_method = split_method

    def run_cv(self, X, Y):
        """

        :param X:
        :param Y:
        :param split_method: iterator object for data splits.
        Choose among
        (KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit)
        :return: array of scores
        """
        cv_scores = []
        splits = self.split_method(self.n_splits)
        for tv_index, val_index in splits.split(X):
            X_tv, Y_tv = X[tv_index], Y[tv_index]
            X_val, Y_val = X[val_index], Y[val_index]
            cv_scores.append(self.model.test_score(X_tv, Y_tv, X_val, Y_val, self.score_criteria))
        return cv_scores


class LGPC_GridSearchCV:
    """
    Run GridSearch with CV for LPGC
    """
    def __init__(self, hyp_test, param_grid, score_criteria=roc_auc_score, n_splits=5, split_method=StratifiedKFold):
        """
        :param hyp_test: str - type of hypothesis testing to do (ie. score, t, ...)
        :param param_grid: dict() for parameters
        :param score_criteria: object for scoring criteria ie) roc_auc_score
        :param n_splits: numbero f splits for CV
        :param split_method: onject for splitting data. Chose among
        (KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit)
        """
        self.hyp_test_type = hyp_test
        self.param_grid = param_grid
        # ex)
        # TODO: how to exactly match kernel type with kernel hyperparameters?
        # param_grid = {
        #     "kernel_hyperparams": [[], []],
        #     "kernel_type": ["Matern", "RationalQuadratic", "ExpSineSquared"],
        #     "k": [50, 100, 150, 200]
        # }
        self.score_criteria = score_criteria
        self.n_splits = n_splits
        self.split_method = split_method

    def run(self, X, Y):
        scores = dict()

        # Run CV on all combinations & save scores
        for kernel_type in self.param_grid["kernel_type"]:
            for kernel_hyperparams in self.param_grid["kernel_hyperparams"]:
                for k in self.param_grid["k"]:
                    model = LocalGaussianProcessClassifier(kernel_hyperparams=kernel_hyperparams,
                                                           kernel_type=kernel_type, k=k)

                    scores[(kernel_type, kernel_hyperparams, k)] = \
                        LGPC_CV(model=model, score_criteria=self.score_criteria,
                                n_splits=self.n_splits, split_method=self.split_method).run_cv(X, Y)

        # TODO: Write selection method for optimal hyperparameter
        # do sth with scores to achieve the above
        optimal_hyperparams = {
            "kernel_hyperparams": self.param_grid["kernel_hyperparams"][0],
            "kernel_type": self.param_grid["kernel_type"][0],
            "k": self.param_grid["k"][0]
        }
        return optimal_hyperparams

    def run_raw(self, X, Y):
        """
        returns the dictionary for scores insead of of optimal hyperparameters
        :param X:
        :param Y:
        :return: returns the dictionary for scores
        """
        scores = dict()

        # Run CV on all combinations & save scores
        for kernel_type in self.param_grid["kernel_type"]:
            for kernel_hyperparams in self.param_grid["kernel_hyperparams"]:
                for k in self.param_grid["k"]:
                    model = LocalGaussianProcessClassifier(kernel_hyperparams=kernel_hyperparams,
                                                           kernel_type=kernel_type, k=k)

                    scores[(kernel_type, kernel_hyperparams, k)] = \
                        LGPC_CV(model=model, score_criteria=self.score_criteria,
                                n_splits=self.n_splits, split_method=self.split_method).run_cv(X, Y)
        return scores

# example
n = 100  # number of data points
d = 5  # dimensions
X = 100*np.random.random((n, d))-50
Y = np.sign(np.sign(2*np.random.random((n, 1))-1)+0.1)  # y in {-1, +1}

# Model Building and Predicting
X_test = 100*np.random.random((70, d))-50  # test data
model = LocalGaussianProcessClassifier(kernel_hyperparams=[1, (0.01, 5), 1.5], kernel_type="Matern", k=25)
Y_test_predictions = model.predict(X_tr=X, Y_tr=Y, X_te=X_test)
predictions = model.predict(X_tr=X, Y_tr=Y, X_te=X_test)
training_score = model.score(X_tr=X, Y_tr=Y, score_criteria=roc_auc_score)

# CV
CV = LGPC_CV(model=model, score_criteria=roc_auc_score, n_splits=5, split_method=StratifiedKFold)
cv_scores = CV.run_cv(X=X, Y=Y)

# GridSearch CV
param_grid = {
    "kernel_hyperparams": [[1.0, (0.001, 10), 0.5], [1.0, (0.01, 10), 2.5], [2.0, (0.01, 5), float("inf")]],
    "kernel_type": ["Matern"],
    "k": [100, 150]
}
grid_search_cv = LGPC_GridSearchCV(hyp_test="t", param_grid=param_grid,
                                   score_criteria=roc_auc_score, n_splits=5, split_method=StratifiedKFold)
scores_dict = grid_search_cv.run_raw(X, Y)
optimal_hyperparameters = grid_search_cv.run(X, Y)

