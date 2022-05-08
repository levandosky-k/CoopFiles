import numpy as np
import matplotlib.pyplot as plt

def graph(filename, savename, bins, steps):
    y=np.genfromtxt(filename)
    x=np.linspace(-4,4,bins)
    i=0
    while i<steps:
        plt.plot(x,y[i])
        i=i+10
    plt.savefig(savename)
