import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

'''
Calculates the normal vector of the hyperplane

the normal vector is the theta input

theta input should be 

'''


def DRAWHyperplane(theta, theta_zero):
    # normal to hypeplane line is #y = mx + c
    # or 0 = (theta_1 * x_1) + (theta_2 * x_2) + theta_zero

    # where theta_2 part is effectively our Y value in y = mx+c
    # theta_zero is the c, and theta_1 bit is our x
    X_test = [np.linspace(-10, 10) * theta[0], np.linspace(-10, 10) * theta[1]]
    hyperplane = [np.linspace(-10, 10) * theta[1], (np.linspace(-10, 10) * theta[0] * -1) + theta_zero]
    plt.plot(X_test[0], X_test[1], color='blue', linestyle='--')
    plt.axhline(0, linestyle='solid', color='black')
    plt.axvline(0, linestyle='solid', color='black')
    plt.axis('equal')
    plt.plot(hyperplane[0], hyperplane[1], color='red', linestyle='solid')
    plt.show()


'''
x is point on the hyperplane


'''


class Hyperplane:
    def __init__(self, theta, theta_zero, x_hyperplane=None):
        self.theta = np.array(theta)
        self.hyperplaneform = np.array([self.theta[1], self.theta[0] * -1])
        if theta_zero is not None:
            self.theta_zero = theta_zero
        else:
            if x_hyperplane is None:
                raise Exception("Both theta_zero and a point on hyperplane cant be none")
            self.theta_zero = self.calcCfromgradientandPoint(self.hyperplaneform,x_hyperplane)


    def calcCfromgradientandPoint(self,grad,x):
        return x[1] - (grad[1] / grad[0] * x[0] )
    def getX2onHyperplane(self,X1):
        return -1 * ((self.theta[0] * x) + self.theta_zero) / self.theta[1]

    def getX1onHyperplane(self,X2):
        return self.hyperplaneform[0] * (X2 - self.theta_zero) / self.hyperplaneform[1]

    def getX2onNormal(self,X1):
        return self.theta[1] / self.theta[0] * X1
    def getX1onNormal(self,X2):
        return X2 * self.theta[0] / self.theta[1]

    def drawHyperplane(self,point=None):
        normal = [np.linspace(-10, 10) * self.theta[0], np.linspace(-10, 10) * self.theta[1]]
        hyperplane =  [np.linspace(-10, 10) * self.theta[1], (np.linspace(-10, 10) * self.theta[0] * -1) + self.theta_zero]
        plt.plot(normal[0], normal[1], color='blue', linestyle='--',label="normal")
        plt.axhline(0, linestyle='solid', color='black')
        plt.axvline(0, linestyle='solid', color='black')
        plt.axis('equal')
        plt.plot(hyperplane[0], hyperplane[1], color='red', linestyle='solid',label="hyperplane")
        if point is not None:
            plt.axhline(point[1], linestyle=':', color='green')
            plt.axvline(point[0], linestyle=':', color='green')
        plt.legend(loc="upper left")
        plt.show()


test = Hyperplane([1,2],-1.5)
print(test.hyperplaneform)
#test.drawHyperplane()

test2 = Hyperplane([1,2],theta_zero=None,x_hyperplane=[0,-1.5])
print(test2.hyperplaneform)
#test2.drawHyperplane()

print("testing get X2")
print(test.theta_zero)
#print(test.getX2onHyperplane(10))
print(test2.getX2onHyperplane(10))
#test2.drawHyperplane()
#test2.drawHyperplane([10,test2.getX2onHyperplane(10)])
print("checking getX1")
print(test2.getX1onHyperplane(-6.5))

print("checking getXY on normal")

print(test2.getX2onNormal(0))
print(test2.getX1onNormal(0))
test2.drawHyperplane([0,-0.75])
