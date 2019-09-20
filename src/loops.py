import numpy as np

import calculbulles as cb

# import pstats
# import cProfile


class Loops():
    __KEYS = set(["Np", "tp2", "tp", "Ef"])

    def __init__(self, parameters):
        parameter_keys = set(parameters.keys())
        miss = []
        for pa in Loops.__KEYS:
            if pa not in parameter_keys:
                miss.append(pa)
        miss.sort()
        if Loops.__KEYS.issubset(parameter_keys):
            self.param = parameters
        else:
            message = "{} must have {} keys".format(
                parameters.__repr__(), ','.join(miss))
            raise KeyError(message)
        try:
            Np = int(self.param["Np"])
        except KeyError:
            Np = int(self.param["N_patche"])

        if "T" in self.param:
            self.Temperature = float(self.param["T"])

        try:
            self.Temperature = float(self.param["Temperature"])
        except KeyError:
            self.param["Temperature"] = 1e-80
        finally:
            self.Temperature = 1e-80

        for k in parameter_keys:
            if k not in Loops.__KEYS and not k.startswith("T"):
                del self.param[k]

        self.Cooper = np.zeros((Np, Np, Np), float)
        self.Peierls = np.zeros((Np, Np, Np), float)
        self.Peierls_susc = np.zeros((Np, 2), float)

    def initialize(self, Temperature=1e-80, **kwargs):
        '''
            Set Loops values to null.

        '''
        self.Temperature = Temperature
        for key, value in kwargs.items():
            self.param[key] = value

        self.Cooper = np.zeros(self.Cooper.shape, float)
        self.Peierls = np.zeros(self.Peierls.shape, float)
        self.Peierls_susc = np.zeros(self.Peierls_susc.shape, float)

    def get_values(self, l_rg: float):

        cb.valeursbulles(l_rg, self.Temperature, self.param, self.Cooper,
                         self.Peierls, self.Peierls_susc)

    def __call__(self, l_rg: float):
        '''
            Calculate the values of Cooper and Peierls loops
            for a given Temperature and RG_l parameter.
            This is donne via cbulle module using cython
            to do integration for time considerations.

        '''
        try:
            self.param["Temperature"] = self.Temperature

            cb.valeursbulles(l_rg, self.Temperature, self.param, self.Cooper,
                             self.Peierls, self.Peierls_susc)
            del self.param["Temperature"]
        except Exception as identifier:
            print(identifier)
            return False
        else:
            return True

        return True
