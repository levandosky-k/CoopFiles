import numpy as np

array=[]
i=1
j=1
while i <= 431:
    file=open('PlotValues%i.csv' % i, 'rb')
    next(file)
    data=np.loadtxt(file, delimiter=',')
    file.close()
    if i==1:
        array=np.zeros((87,len(data[:,1])))
    array[j-1,:]=data[:,1]
    i=i+5
    j=j+1
print(array)
np.savetxt('dye_data_1sec.txt', array)
