import numpy as np
import matplotlib.pyplot as plt

#def estimater_rosenblatt(x):
#        return np.count_nonzero((data < x+h)&(x-h < data))/(size*h)


class Estimator:
    def __init__(self, data, bandwidth, split_num):
        self.data = data
        self.h = bandwidth
        self.sp_num = split_num
        
# rosenblatt 
    def rosenblattKernel(self, x):
        data = self.data
        h = self.h
        return np.count_nonzero((data < x+h)&(x-h < data))/(2*len(data)*h)
    
    def rosenblattPlot(self):
        d = self.data
        p = np.linspace( min(d), max(d), self.sp_num) #this 100 would be changed
        vfunc = np.vectorize(self.rosenblattKernel)
        result = vfunc(p)
        plt.plot(p, result, color = "m", label = "rosenblatt estimator")
        plt.legend()
        print(sum(result))


    def triangleKernel(self, x):
        data = self.data
        h = self.h
        return np.array([(1- np.abs((x-X)/h))*((x < X + h)&(X - h < x) ) for X in data]).sum()/(len(data)*h)

    def trianglePlot(self):
        d = self.data
        p = np.linspace( min(d), max(d), self.sp_num) 
        vfunc = np.vectorize(self.triangleKernel)
        result = vfunc(p)
        plt.plot(p, result, color = "m", label = "triangle estimator")
        plt.legend()
        print(sum(result)*(max(self.data)- min(self.data))/self.sp_num)

    def paraboricKernel(self, x):
        data = self.data
        h = self.h
        return np.array([(1- ((x-X)/h)**2)*((x < X + h)&(X - h < x) ) for X in data]).sum()/(len(data)*h)*3/4
    
    def paraboricPlot(self):
        d = self.data
        p = np.linspace( min(d), max(d), self.sp_num) 
        vfunc = np.vectorize(self.paraboricKernel)
        result = vfunc(p)
        plt.plot(p, result, color = "m", label = "paraboric estimator")
        plt.legend()
        print(sum(result)*(max(self.data)- min(self.data))/self.sp_num)

    
    def biweightKernel(self, x):
        data = self.data
        h = self.h
        return np.array([((1- ((x-X)/h)**2)**2)*((x < X + h)&(X - h < x) ) for X in data]).sum()/(len(data)*h)*15/16


    def biweightPlot(self):
        d = self.data
        p = np.linspace( min(d), max(d), self.sp_num) 
        vfunc = np.vectorize(self.biweightKernel)
        result = vfunc(p)
        plt.plot(p, result, color = "m", label = "biweight estimator")
        plt.legend()
        print(sum(result)*(max(self.data)- min(self.data))/self.sp_num)    


    def gaussianKernel(self, x):
        data = self.data
        h = self.h
        return np.array([np.exp( -1*((x-X)/h)**2/2 ) for X in data]).sum()/(len(data)*h)*np.sqrt(1/(2*np.pi))
    

    def gaussianPlot(self):
        d = self.data
        p = np.linspace( min(d), max(d), self.sp_num) 
        vfunc = np.vectorize(self.gaussianKernel)
        result = vfunc(p)
        plt.plot(p, result, color = "m", label = "gaussian estimator")
        plt.legend()
        print(sum(result)*(max(self.data)- min(self.data))/self.sp_num)    
    
    
    def silvermanKernel(self, x):
        data = self.data
        h = self.h
        return np.array([np.exp( -1*np.abs((x-X)/h)/np.sqrt(2) )*np.sin( np.abs((x-X)/h)/np.sqrt(2) + np.pi/4 ) for X in data]).sum()/(len(data)*h)/2

    
    def silvermanPlot(self):
        d = self.data
        p = np.linspace( min(d), max(d), self.sp_num) 
        vfunc = np.vectorize(self.silvermanKernel)
        result = vfunc(p)
        plt.plot(p, result, color = "m", label = "silverman estimator")
        plt.legend()
        print(sum(result)*(max(self.data)- min(self.data))/self.sp_num)


    def Error(self, valueAt0):
        return valueAt0 - self.silvermanKernel(0)
