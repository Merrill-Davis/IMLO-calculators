import numpy as np

def signed_dist(x,theta,theta_0):
    print(f"Signed perpendicular distance: {(theta.T@x + theta_0) / np.linalg.norm(theta)}")
    return (theta.T@x + theta_0) / np.linalg.norm(theta)


x = np.array([4,-0.5])
x2 = np.array([0,0])
theta = np.array([3,4])
theta_0 = 5

print(signed_dist(x,theta,theta_0))
signed_dist(x2,theta,theta_0)