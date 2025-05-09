import numpy as np
import math
x_train = np.array([1.0, 2.0])  
y_train = np.array([300.0, 500.0])
def compute_cost(x,y,w,b):
    m = x.shape[0]
    j_wb = 0
    for i in range(m):
        f_wb = w * x[i] + b
        lost = (f_wb - y[i]) ** 2
        j_wb = 1/ (2*m) * lost
    return j_wb
def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_db += 1/m * dj_db_i
        dj_dw += 1/m * dj_dw_i
    return dj_dw, dj_db
def gradient_descent(x,y,w_in,b_in,alpha,num_iters, cost_func,gradient_func):
    j_history = []
    p_history = []
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            j_history.append(cost_func(x,y,w,b))
            p_history.append([w,b])
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w,b,j_history, p_history
w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")



