import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = 1/3 if feature_subsample_size is None else feature_subsample_size
        self.models = [DecisionTreeRegressor(max_depth=max_depth, **trees_parameters) for _ in range(n_estimators)]

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        n_subspace = int(X.shape[0] * (1 - 1/np.e))
        n_features = int(X.shape[1] * self.feature_subsample_size)
        self.features_num = []
        for model in self.models:
            subspace = np.random.choice(X.shape[0], n_subspace, replace=False)
            features = np.random.choice(X.shape[1], n_features, replace=False)
            self.features_num.append(features)
            model.fit(X[subspace][:, features], y[subspace])

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        regressor_ans = 0
        for model, features in zip(self.models, self.features_num):
            regressor_ans += model.predict(X[:, features])
        return regressor_ans / self.n_estimators


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = 1/3 if feature_subsample_size is None else feature_subsample_size
        self.models = [DecisionTreeRegressor(max_depth=max_depth, **trees_parameters) for _ in range(n_estimators)]

    def predict_on_estimators(self, X, estimators):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        estimators : List[estimator]
            Array of boosting estimators in ensamble
        """
        answers = np.zeros(X.shape[0], dtype=np.float32)
        for estimator, features, coef in zip(estimators, self.features_num, self.coef):
            answers += coef * estimator.predict(X[:, features])
        return answers

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        n_subspace = int(X.shape[0] * (1 - 1/np.e))
        n_features = int(X.shape[1] * self.feature_subsample_size)
        self.features_num = []
        self.coef = []
        gradient = 2 * y
        y_predicted = np.zeros(X.shape[0])
        for i, model in enumerate(self.models):
            subspace = np.random.choice(X.shape[0], n_subspace, replace=False)
            features = np.random.choice(X.shape[1], n_features, replace=False)
            self.features_num.append(features)
            model.fit(X[subspace][:, features], gradient[subspace])
            y_pred = model.predict(X[:, features])
            loss = lambda alpha: np.mean(((y_predicted + alpha * self.learning_rate * y_pred) - y) ** 2)
            best_alpha = minimize_scalar(loss)
            self.coef.append(best_alpha.x * self.learning_rate)
            y_predicted += best_alpha.x * self.learning_rate * y_pred
            gradient = 2 * (y - y_predicted)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        return self.predict_on_estimators(X, self.models)
