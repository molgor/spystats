
from spystats.statistical import *
t_30_10 = exponentialModelFunctional(points,phi=30,sigma=10)
phis =np.linspace(20,40,20)
sigmas =np.linspace(5,15,10)
phis_sigma = [(phi , sigma) for phi in phis for sigma in sigmas]
ys = t_30_10.rvs()
superfunciones = map(lambda (phi,sigma) : likelihoodFunction(phi,sigma,ys,points),phis_sigma)
A = np.array(superfunciones).reshape(10,20)
import matplotlib.pyplot as plt
plt.imshow(A)
plt.show()
A = np.array(superfunciones).reshape(20,10)
plt.imshow(A)
plt.show()
superfunciones
chula = zip(superfunciones,phis_sigma)
chula
chula.sort(key=lambda renglon : renglon[0])








