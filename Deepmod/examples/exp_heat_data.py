import numpy as np
import torch

#all terms

def exp_heat_delta(x: torch.tensor, t: torch.tensor, D: float):
    #copied from deepymod/data/burgers/burgers_delta
    x=torch.linspace(-4,4,801)
    t=torch.linspace(0,1,39)
    t,x = torch.meshgrid(t,x) #meshgrid creates a grid of coordinates
    
    u= 255-torch.Tensor(np.genfromtxt("dye_data_green.txt"))
    #u=u[0:2,:]
    u=u/255
    #newu=torch.zeros_like(u)
    #newu=torch.cat((newu,newu), dim=0)
    #newu=torch.cat((newu,newu),dim=0)
    #i=0
    #while i<156:
    #    newu[i]=u[38]
    #    i=i+1
    #i=1
    #start=0
    #stop=800
    #while i<156:
    #    j=0
    #    while j<420:
    #        newu[i,j]=newu[i-1,j+1]-0.001
    #        j=j+1
    #    while j<801:
    #        newu[i,j]=newu[i-1,j-1]-0.001
    #        j=j+1
        #newu[i,410:430]=newu[i,410:430]-0.001
    #    i=i+1
    #u=torch.cat((u,newu),dim=0)
    #np.savetxt("modified_data.txt", u)
    #u[u<0.25]=0
    #u=np.transpose(u)
    coords = torch.cat((t.reshape(-1, 1), x.reshape(-1, 1)), dim=1)
    #print(coords.shape)
    #print(u.view(-1,1).shape)
    #print(u.shape)
    return coords, u.view(-1, 1)

