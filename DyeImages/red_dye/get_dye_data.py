import numpy as np

array=[]
i=1 
while i <= 18:
    file=open('PlotValues%ib.csv' % i, 'rb')
    next(file)
    data=np.loadtxt(file, delimiter=',')
    file.close()
    if i==1:
        array=np.zeros((18,len(data[:,1])))
    array[i-1,:]=data[:,1]
    i=i+1
print(array)
np.savetxt('dye_data2.txt', array)
