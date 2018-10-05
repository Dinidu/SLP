from numpy import genfromtxt

class Normalize(object):
    def __init__(self,file):
        self.data= genfromtxt(file, delimiter=',')
        self.xmin = 0
        self.xmax = 0

    def set_min_max(self):
        self.data = [1,2,3]
        for data_el in self.data:
            self.xmax = max(data_el)
            self.xmin = min(data_el)

    @staticmethod
    def simple(self,xi):
        z = (xi - self.xmin)/(self.xmax - self.xmin)
        return z

    def prepare(self):
        data_set = genfromtxt('input.csv', delimiter=',')





