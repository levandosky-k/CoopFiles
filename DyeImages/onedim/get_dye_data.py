import numpy as np

array=[]
i=6971
j=1
while i <= 7030:
    if i < 7000 or i > 7001:
        file=open('PlotValues%i.csv' % i, 'rb')
        next(file)
        data=np.loadtxt(file, delimiter=',')
        file.close()
        if j==1:
            array=np.zeros((58,len(data[:,1])))
        array[j-1,:]=data[:,1]
        i=i+1
        j=j+1
    else:
        i=i+1
print(array)
np.savetxt('dye_data_onedim.txt', array)
