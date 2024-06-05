import numpy as np

def signed_dist(x,theta,theta_0):
    print(f"Signed perpendicular distance: {(theta.T@x + theta_0) / np.linalg.norm(theta)}")
    return (theta.T@x + theta_0) / np.linalg.norm(theta)


x = np.array([-2, 2])
x2 = np.array([0,0])
x3 = np.array([3,-1])
theta = np.array([2,0])
theta_0 = -3

print(signed_dist(x,theta,theta_0))
signed_dist(x2,theta,theta_0)
signed_dist(x3,theta,theta_0)