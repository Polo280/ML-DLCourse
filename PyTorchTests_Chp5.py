# Code tests in Chapter 5 --> The mechanics of learning
# Jorge 07/10/23

import torch
import math
from matplotlib import pyplot as plt

def model(t_u, w, b):
    # W = Weight, b = bias
    return w * t_u + b  # Equation of a line

def loss_calculator(t_real: torch.Tensor, t_model: torch.Tensor):  # Mean square error
    squared_diffs = (t_real - t_model) ** 2
    return squared_diffs.mean()

def loss_derivative(t_real: torch.Tensor, t_model: torch.Tensor):   # x^2 --> dx = 2x
    sq_diff_derivative = 2 * (t_model - t_real) / t_model.size(0)
    return sq_diff_derivative

def dmodel_dw(t_u, w, b):   # Partial derivative of the model with respect to w
    return t_u

def dmodel_db(t_u, w, b):   # Partial derivative of the model with respect to b
    return 1.0

def get_model_gradient(t_real, t_data, t_model, w, b):
    dloss_dmodel = loss_derivative(t_real, t_model)       # Derivative of the loss to the model
    dloss_dw = dloss_dmodel * dmodel_dw(t_data, w, b)     # Compute derivatives of loss with respect to each variable
    dloss_db = dloss_dmodel * dmodel_db(t_data, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

def training_loop(num_epochs: int, learning_rate : float, params : torch.tensor, t_real, t_data):
    for epoch in range(num_epochs + 1):
        w, b = params
        t_model = model(t_data, w, b)
        loss = loss_calculator(t_real, t_model)
        gradient = get_model_gradient(t_real, t_data, t_model, w, b)
        params = params - learning_rate * gradient   # Update parameters accordingly
        if(epoch % 500 == 0):
            print("Epoch {}, Loss {}".format(epoch, loss))
    return params

def main():
    # Define values
    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]  # Temp in Celsius
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]  # Temp in unknown units
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
    w = torch.ones(())
    b = torch.zeros(())

    t_u *= 0.1   # Normalize the tensor since its gradient is so much larger than the gradient of b
    params = training_loop(num_epochs=5000, learning_rate=(math.e.__pow__(-4)), params=torch.tensor([1.0, 0.0]), t_real=t_c, t_data=t_u)
    print(params)

    #Plot
    t_p = model(t_u, *params)
    fig = plt.figure(dpi=600)
    plt.xlabel("Fahrenheit °F")
    plt.ylabel("Celsius °C")
    plt.plot(t_u.numpy(), t_p.detach().numpy())
    plt.plot(t_u.numpy(), t_c.numpy(), 'o')

    # the following code is to do it without using algebraic derivative, instead you use a small change called delta.
    '''
    t_p = model(t_u, w, b)  # Vector with model in current iteration
    loss = loss_calculator(t_c, t_u)
    # Loss rates of change
    delta = 0.1
    loss_roc_w = (loss_calculator(t_c, model(t_u, w + delta, b)) - loss_calculator(t_c, model(t_u, w - delta, b))) / (2 * delta)
    loss_roc_b = (loss_calculator(t_c, model(t_u, w, b + delta)) - loss_calculator(t_c, model(t_u, w, b - delta))) / (2 * delta)
    # Use learning rate to update
    learning_rate = torch.e ** (-2)
    w = w - learning_rate * loss_roc_w
    b = b - learning_rate * loss_roc_b
    '''

main()
