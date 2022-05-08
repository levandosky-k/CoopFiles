import numpy as np
import torch


#def get_heat_data():
#    return data_object


def heat_delta(x: torch.tensor, t: torch.tensor, D: float):
    #copied from deepymod/data/burgers/burgers_delta
    x, t = torch.meshgrid(x, t) #meshgrid creates a grid of coordinates
    

    u = (1/((4*np.pi*t*D)**(1/2))*np.exp(-(x**2)/(4*t))) #* (D ** (1/2))
    coords = torch.cat((t.reshape(-1, 1), x.reshape(-1, 1)), dim=1)
    return coords, u.view(-1, 1)



# want to look like:
# array({'u': array([[ ......numbers......],[...numbers...],......]]),
#           't': array([[...]])
#           'x': array([[...]])     }, dtype=object)
