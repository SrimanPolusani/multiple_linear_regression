# Import statements
import numpy as np
import math
import matplotlib.pyplot as plt


def convertor(txt_data: str, x_usecols: tuple):
    X_train = np.loadtxt(txt_data, usecols=x_usecols, delimiter=',')
    y_train = np.loadtxt(txt_data, usecols=x_usecols[-1] + 1, delimiter=',')
    return X_train, y_train


class multiple_linear_regressor:
    def __init__(self, X_train, X_feature_names, y_train, w_init, b_init, alpha, no_iters, normalization_type: str):
        self.features = X_train
        self.targets = y_train
        self.features_list = X_feature_names
        self.mean = np.mean(X_train, axis=0)
        self.sigma = np.std(X_train, axis=0)
        self.default_features = self.desired_normalization(normalization_type)
        self.w = w_init
        self.b = b_init
        self.learning_rate = alpha
        self.total_iters = no_iters
        self.m, self.n = X_train.shape  # Calculating m (training ex) and n (no of features)
        self.correct_w = None
        self.correct_b = None

    def desired_normalization(self, normalization):
        if normalization == 'mn':
            return self.mean_norm(self.features)
        elif normalization == 'zn':
            return self.zscore_norm(self.features)

    # Calculating cost
    def find_cost(self):
        cost = 0
        for i in range(self.m):  # Calculating f_x
            f_wi = np.dot(self.features[i], self.w) + self.b
            cost = cost + (f_wi - self.targets[i]) ** 2
        cost = cost / (2 * self.m)
        return cost

    # Calculate dJ/dw & dJ/db (gradient)
    def calculate_gradient(self):
        dj_dw = np.zeros(self.n)
        dj_db = 0
        for i in range(self.m):
            f_xwb = np.dot(self.w, self.features[i]) + self.b
            err = f_xwb - self.targets[i]
            for j in range(self.n):
                dj_dw[j] = dj_dw[j] + (err * self.features[i, j])
            dj_db = dj_db + err
        dj_dw = dj_dw / self.m
        dj_db = dj_db / self.m
        return dj_dw, dj_db

    # Perform the gradient descent
    def perform_gradient_descent(self):
        j_his = []
        w_his = []
        for iter in range(self.total_iters):
            dj_dw, dj_db = self.calculate_gradient()
            self.w = self.w - (self.learning_rate * dj_dw)
            self.b = self.b - (self.learning_rate * dj_db)
            if iter < 10000:
                j_his.append(self.find_cost())
            if iter % math.ceil(self.total_iters / 10) == 0:
                w_his.append(self.w)
                print(self.find_cost())
                # print("Interation number: {}, The current w value is {}".format(iter, self.w))

        self.correct_w = self.w
        self.correct_b = self.b
        return self.correct_w, self.correct_b, j_his, w_his

    def mean_norm(self, X_train):  # Mean normalization
        mean_nor = (X_train - self.mean) / (X_train.max(axis=0) - X_train.min(axis=0))
        return mean_nor

    def zscore_norm(self, X_train):  # Zscore normalization
        zscore_nor = (X_train - self.mean) / self.sigma
        return zscore_nor

    @staticmethod
    def make_prediction(X_train, w, b):
        prediction = np.dot(X_train, w) + b
        print("prediction: {}".format(prediction))
        return prediction

    def accuracy_visualization(self, w, b):
        f_xwb = np.dot(self.features, w) + b
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
        for i in range(len(axes)):
            for j in range((len(axes))):
                axes[i].scatter(self.features[:, i], self.targets)
                axes[i].scatter(self.features[:, i], f_xwb)
                axes[i].set_xlabel(self.features_list[i])
                axes[i].set_ylabel("Price (1000s)")
        plt.show()


X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
b_init = 7
w_init = np.array([3, 18, 53, 26])

X_train, y_train = convertor('./houses.txt', x_usecols=(0, 1, 2, 3))

final_model = multiple_linear_regressor(
    X_train,
    X_features,
    y_train,
    np.zeros_like(w_init),
    0,
    5.0e-7,
    1000,
    'zn'
)

w, b, _, _ = final_model.perform_gradient_descent()
print(w, b)
final_model.accuracy_visualization(w, b)