import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools


def linear_classify(x, theta, theta_0):
    """Uses the given theta, theta_0, to linearly classify the given data x. This is our hypothesis or hypothesis class.

    :param x: input data
    :param theta: vector of norm
    :param theta_0: offset
    :return: 1 if the given x is classified as positive, -1 if it is negative, and 0 if it lies on the hyperplane.
    """
    # Todo: Implement the linear classifier here that classifies x given theta, theta_0, and returns the result.
    return np.sign(theta.T @ x + theta_0)


def Loss(prediction, actual):
    """Computes the loss between the given prediction and actual values.

    :param prediction:
    :param actual:
    :return:
    """
    # Todo: Implement the loss between a predicted and actual value here, and return the loss.
    if prediction == actual:
        return 0
    else:
        return 1


def E_n(h, data, labels, L, theta, theta_0):
    """Computes the error for the given data using the given hypothesis and loss.

    :param h: Hypothesis class, for example a linear classifier.
    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param L: A loss function to compute the error between the prediction and label.
    :param theta:
    :param theta_0:
    :return:
    """
    (d, n) = data.shape
    # Todo: Compute the training loss E_n here and return it.
    loss = 0
    for i in range(n):
        x = data[:, i-1]
        y = labels[:, i-1]
        y_hat = h(x,theta,theta_0)
        err = L(y_hat,y)
        loss = loss + err
    return loss / n

def random_linear_classifier(data, labels, params={}, hook=None):
    """

    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param params: A dict, containing a key T, which is a positive integer number of steps to run
    :param hook: An optional hook function that is called in each iteration of the algorithm.
    :return:
    """
    k = params.get('k', 100)  # if k is not in params, default to 100
    (d, n) = data.shape

    # Todo: Implement the Random Linear Classifier learning algorithm here.
    # Note: To call the hook function, use the following line inside your training loop:
    #   if hook: hook((theta, theta_0))
    lowest_loss = 1E100
    best_theta = None
    best_theta_0 = None

    for i in range(k):
        theta = np.random.uniform(-20,20,(d, 1))
        theta_0 = np.random.uniform(-20,20)
        loss = E_n(linear_classify,data,labels,Loss,theta,theta_0)
        if loss < lowest_loss:
            lowest_loss = loss
            best_theta = theta
            best_theta_0 = theta_0
        if hook:
            hook((theta, theta_0))
    return (best_theta,best_theta_0)


def perceptron_with_offset(data, labels, params={}, hook=None):
    """The Perceptron learning algorithm.

    :param data: A d x n matrix where d is the number of data dimensions and n the number of examples.
    :param labels: A 1 x n matrix with the label (actual value) for each data point.
    :param params: A dict, containing a key T, which is a positive integer number of steps to run
    :param hook: An optional hook function that is called in each iteration of the algorithm.
    :return:
    """
    T = params.get('T', 100)  # if T is not in params, default to 100
    (d, n) = data.shape

    # Todo: Implement the Perceptron algorithm here.
    theta = np.zeros(d)
    theta_0 = 0
    for t in range(T):
        for i in range(n):
            x = data[:, i:i+1]
            y = labels[:, i:i+1]
            if y * (theta.T @ x + theta_0) <= 0:
                theta = theta + (y * x).flatten()
                theta_0 = theta_0 + y
        if hook:
            hook((theta,theta_0))
    return theta,theta_0


def plot_separator(plot_axes, theta, theta_0,points):
    """Plots the linear separator defined by theta, theta_0, into the given plot_axes.

    :param plot_axes: Matplotlib Axes object
    :param theta:
    :param theta_0:
    """

    # One way we can plot the intercept is to compute the parametric line equation from the implicit form.
    # compute the y-intercept by setting x1 = 0 and then solving for x2:
    y_intercept = -theta_0 / theta[1]
    # compute the slope (-theta[0]/theta[1], I think)
    slope = -theta[0] / theta[1]
    # Then compute two points using:
    xmin, xmax = -15, 15
    xmin = min(points[0]) - 15
    xmax = max(points[0]) + 15
    # Note: It's not ideal to only plot the lines in a fixed region, but it makes this code simple for now.

    p1_y = slope * xmin + y_intercept
    p2_y = slope * xmax + y_intercept


    # Plot the separator:
    plot_axes.plot([xmin, xmax], [p1_y.flatten()[0], p2_y.flatten()[0]], '-')

    # hyperplane = [np.linspace(xmin, xmax),
    #               [-1 * ((theta[0]*i) + theta_0[0][0]) / theta[1] for i
    #                in np.linspace(xmin, xmax)]
    #               ]


    #plot_axes.plot(hyperplane[0], hyperplane[1], color='blue', linestyle='solid',label="hyperplane")
    # Plot the normal:
    # Note: The normal might not appear perpendicular on the plot if ax.axis('equal') is not set - but it is
    # perpendicular. Resize the plot window to equal axes to verify.
    plot_axes.arrow((xmin + xmax) / 2, (p1_y.flatten()[0] + p2_y.flatten()[0]) / 2, float(theta[0]), float(theta[1]))


def transform_polynomial_basis(x, order):

    if order == 0:
        return [1]
    else:
        basis = transform_polynomial_basis(x, order-1)

        for pair in itertools.product(*[x for _ in range(order)]):
            basis.append(np.prod(pair))
        return list(set(basis))


if __name__ == '__main__':
    """
    We'll define data X with its labels y, plot the data, and then run either the random_linear_classifier or the
    perceptron learning algorithm, to find a hypothesis h from the class of linear classifiers.
    We then plot the best hypothesis, as well as compute the training error. 
    """

    # Let's create some training data and labels:
    #   X is a d x n matrix where d is the number of data dimensions and n the number of examples. So each data point
    #     is a column vector.
    #   y is a 1 x n matrix with the label (actual value) for each data point.
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])

    # To test your algorithm on a larger dataset, uncomment the following code. It generates uniformly distributed
    # random data in 2D, along with their labels.
    # X = np.random.uniform(low=-5, high=5, size=(2, 20))  # d=2, n=20
    # y = np.sign(np.dot(np.transpose([[3], [4]]), X) + 6)  # theta=[3, 4], theta_0=6

    # Plot positive data green, negative data red:
    colors = np.choose(y > 0, np.transpose(np.array(['r', 'g']))).flatten()
    plt.ion()  # enable matplotlib interactive mode
    fig, ax = plt.subplots()  # create an empty plot and retrieve the 'ax' handle
    ax.scatter(X[0, :], X[1, :], c=colors, marker='o')
    # Set up a pretty 2D plot:
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # ax.set_xlim(-20, 20)
    # ax.set_ylim(-20, 20)
    ax.set_xlim(min(X[0]) - 10,max(X[0]) + 20)
    ax.set_xlim(min(X[1]) - 10, max(X[1]) + 20)
    ax.grid(True, which='both')
    ax.axhline(color='black', linewidth=0.5)
    ax.axvline(color='black', linewidth=0.5)
    ax.set_title("Linear classification")



    # We'll define a hook function that we'll use to plot the separator at each step of the learning algorithm:
    def hook(params):
        (th, th0) = params
        plot_separator(ax, th, th0)


    # Run the RLC or Perceptron: (uncomment the following lines to call the learning algorithms)
    theta, theta_0 = random_linear_classifier(X, y, {"k": 1000}, hook=None)
    theta, theta_0 = perceptron_with_offset(X, y, {"T": 1000}, hook=None)
    # Plot the returned separator:


    # Change to add own theta in

    #theta, theta_0 = [-3,-4],-5




    plot_separator(ax, theta, theta_0,points=X)

    # Run the RLC, plot E_n over various k:
    # Todo: Your code
    plt.show()
    print(f"theta: {theta} , theta_0 : {theta_0}")
    print("Finished.")