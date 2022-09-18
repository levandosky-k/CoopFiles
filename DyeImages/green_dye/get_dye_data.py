import numpy as np

array=[]
i=6211
j=1
while i <= 6249:
    file=open('PlotValuesb%i.csv' % i, 'rb')
    next(file)
    data=np.loadtxt(file, delimiter=',')
    file.close()
    if j==1:
        array=np.zeros((39,len(data[:,1])))
    array[j-1,:]=data[:,1]
    i=i+1
    j=j+1
print(array)
np.savetxt('dye_data_green2.txt', array)
