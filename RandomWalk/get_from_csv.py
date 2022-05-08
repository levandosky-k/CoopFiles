import numpy as np

def get_data(filename, newname):
    file=open(filename, 'rb')
    data=np.loadtxt(file, delimiter=',')
    file.close()
    np.savetxt(newname, np.array(data))
    return np.array(data)
