from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     ShuffleSplit, StratifiedShuffleSplit)
from sklearn.gaussian_process.kernels import (Matern, RationalQuadratic,
                                              ExpSineSquared)
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree


class LocalGaussianProcessClassifier:
    def __init__(self, kernel_hyperparams, kernel_type="Matern", k=100):
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
        # runn k-NN first
        tree = BallTree(X_tr, leaf_size=2)
        dist, ind = tree.query(X_te, k=self.k)
        # GP Calssification
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

    def score(self, X_tr, Y_tr):
        """ Returns AUC for training data
        """
        Y_pred = self.predict(X_tr, X_tr, Y_tr)
        return roc_auc_score(Y_tr, Y_pred)

    def test_score(self, X_tr, Y_tr, X_te, Y_te):
        """ Returns AUC for test data
        """
        Y_pred = self.predict(X_tr, X_te, Y_tr)
        return roc_auc_score(Y_te, Y_pred)


class LGPC_CV:
    """ CV for Local Gaussian Process Calssifier
    """

    def __init__(self, model, k=5):
        """
        k = nubmer of splits (for k-fold cv and shuffle& split CV)
        model = LGPC model
        """
        self.k = k
        self.model = model

    def run_cv(X, Y, split_method):
        """
        split_method = iterator object for data splits. Choose among
            (KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit)
        """
