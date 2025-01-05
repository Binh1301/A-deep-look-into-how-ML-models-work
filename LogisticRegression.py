import numpy as np
def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)
        err = f_wb_i - y[i]
        for j in range(n):

            dj_dw[j] += err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    for j in range(n):
        dj_dw[j] += (lambda_/m) * w[j]
    return dj_db, dj_dw
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )