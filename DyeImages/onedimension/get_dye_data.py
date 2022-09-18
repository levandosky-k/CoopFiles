import numpy as np

array=[]
i=6920
j=1
while i <= 6956:
    file=open('PlotValues%i.csv' % i, 'rb')
    next(file)
    data=np.loadtxt(file, delimiter=',')
    file.close()
    if j==1:
        array=np.zeros((37,len(data[:,1])))
    array[j-1,:]=data[:,1]
    i=i+1
    j=j+1
print(array)
np.savetxt('dye_data_onedim.txt', array)
