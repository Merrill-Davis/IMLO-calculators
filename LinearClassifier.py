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
        #self.hyperplaneform = np.array([self.theta[1], self.theta[0] * -1])
        if theta_zero is not None:
            self.theta_zero = theta_zero
        else:
            if x_hyperplane is None:
                raise Exception("Both theta_zero and a point on hyperplane cant be none")
            self.theta_zero = -1 * ((self.theta[0] * x_hyperplane[0]) + (self.theta[1] * x_hyperplane[1]))

    def getX2onHyperplane(self,X1):
        return -1 * ((self.theta[0] * X1) + self.theta_zero) / self.theta[1]

    def getX1onHyperplane(self,X2):
        return -1 * ((self.theta[1] * X2) + self.theta_zero) / self.theta[0]
        #return self.hyperplaneform[0] * (X2 - self.theta_zero) / self.hyperplaneform[1]

    def drawHyperplane(self,point=None):

        thetasumsquare = pow(self.theta[0], 2) + pow(self.theta[1], 2)
        #
        #
        #startx = min(0,-1 * self.theta_zero * self.theta[0] / thetasumsquare)
        #endx = max(self.theta[0],- 1 * self.theta_zero * self.theta[0] / thetasumsquare)

        # normal = [np.linspace(startx, endx),
        #           [i * self.theta[1] / self.theta[0]
        #            for i in np.linspace(startx, endx)]]

        intersectx = -1 * self.theta_zero * self.theta[0] / thetasumsquare
        intersecty = -1 * self.theta_zero * self.theta[1] / thetasumsquare

        normal = [np.linspace(0, self.theta[0]),
                  [i * self.theta[1] / self.theta[0]
                   for i in np.linspace(0, self.theta[0])]]

        print(f"The normal and the hyperplane intersect at X: {intersectx} Y: {intersecty}")
        #print(f"minimum X: {startx}, endx = {endx}")

        plt.plot(normal[0], normal[1], color='blue', linestyle='--',label="normal")
        plt.axhline(0, linestyle='solid', color='black')
        plt.axvline(0, linestyle='solid', color='black')
        plt.axis('equal')
        start_hypx = -10
        end_hypx = 10
        if point is not None:
            plt.axhline(point[1], linestyle=':', color='green')
            plt.axvline(point[0], linestyle=':', color='green')
            if abs(point[0]) > 10:
                if point[0] >= 0:
                    end_hypx = point[0] + 2
                else:
                    start_hypx = point[0] - 2

        hyperplane = [np.linspace(start_hypx, end_hypx),
                      [self.getX2onHyperplane(i) for i in np.linspace(start_hypx, end_hypx)]]

        plt.plot(hyperplane[0], hyperplane[1], color='red', linestyle='solid',label="hyperplane")

        plt.legend(loc="upper left")
        plt.show()


test = Hyperplane([1,2],1.5)
#test.drawHyperplane()

test2 = Hyperplane([1,2],theta_zero=None,x_hyperplane=[-0.3,-0.6])

print(test.getX2onHyperplane(0))
#test.drawHyperplane()
test.drawHyperplane([-0.3,-0.6])


print(test.getX1onHyperplane(-0.75))
print(test.getX2onHyperplane(0))

print(test2.theta_zero)

test3 = Hyperplane([1,-2],20)
test3.drawHyperplane()

test4 = Hyperplane([-1,-2],20)
test4.drawHyperplane([100,20])