import numpy as np

array=[]
i=6255
j=1
while i <= 6303:
    file=open('PlotValues%i.csv' % i, 'rb')
    next(file)
    data=np.loadtxt(file, delimiter=',')
    file.close()
    if j==1:
        array=np.zeros((49,len(data[:,1])))
    array[j-1,:]=data[:,1]
    i=i+1
    j=j+1
print(array)
np.savetxt('dye_data_coffee.txt', array)
