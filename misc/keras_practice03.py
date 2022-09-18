#practice neural network using pde's
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot


a = 0.5 #delta function constant
D = 0.001 #diffusion constant
num_iters = 10 #number of iterations of time
num_space = 500 #number of values of space (x)


rho = np.zeros((num_iters,num_space)) # num_iters by num_space array

#       | [None, ..., None]|
# rho = |        ...       |
#       | [None, ..., None]|

j=0
while j < num_space:
    rho[0][j] = (1 / (abs(a) * np.sqrt(np.pi))) * jnp.exp(-1 * (j/a) ** 2 )
    j = j+1

i = 1
while i < num_iters:
    rho[i][0] = rho[i-1][0] + (D * (1/num_iters) / (1/num_space)**2) * (rho[i-1][1] - 2*rho[i-1][0])
    rho[i][num_space-1] = rho[i-1][0] + (D * (1/num_iters) / (1/num_space)**2) * (-2*rho[i-1][num_space-1] + rho[i-1][num_space-2])
    j = 1
    while j < num_space - 1:
        rho[i][j] = rho[i-1][0] + (D * (1/num_iters) / (1/num_space)**2) * (rho[i-1][j+1] - 2*rho[i-1][j] + rho[i-1][j-1])
        j = j+1
    i = i+1


#print(rho)




#neural network
for iteration in rho:
    pyplot.plot(np.linspace(0,1,num_space),iteration)
pyplot.show()






