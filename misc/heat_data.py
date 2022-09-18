import numpy as np

N = 100
t = np.linspace(0.5,2,N)
x = np.linspace(-5,5,N)
[xx,tt] = np.meshgrid(x,t)

data_x = xx
data_t = tt
data_fxt = 1/((4*np.pi*tt)**(1/2))*np.exp(-(xx**2)/(4*tt))

data_array = np.array({'u': data_x, 't': data_t, 'x': data_fxt})
data_object = data_array.astype(dtype=object)


def get_heat_data():
    return data_object



# want to look like:
# array({'u': array([[ ......numbers......],[...numbers...],......]]),
#           't': array([[...]])
#           'x': array([[...]])     }, dtype=object)
