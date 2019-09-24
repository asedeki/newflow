
import numpy as np
import difflib
try:
    import LoopsIntegration as cb
except Exception:
    import os
    import sys
    path = os.getcwd().split("/")
    path = "/".join(path[:path.index("newflow")+1])
    sys.path.append(path)
    import lib.LoopsIntegration as cb


# import pstats
# import cProfile

def best_match_key(parameters: dict, loops_dict):
    best_keys = {
        difflib.get_close_matches(key, loops_dict, n=1)[0]: v
        for key, v in parameters.items()
    }
    return best_keys


class Loops():
    __KEYS = set(["Np", "tp2", "tp", "Ef"])
    _instance = None
    loops_integration_donne = False
    Temperature = 1e-80

    # def __new__(cls, *vargs, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(Loops, cls).__new__(cls)
    #     return cls._instance

    def __init__(self, parameters: dict) -> None:
        try:
            Temp = difflib.get_close_matches("T", parameters, n=1)[0]
            self.Temperature = float(parameters[Temp])
        except Exception:
            pass

        matched_parameters = best_match_key(
            parameters, Loops.__KEYS
        )
        miss = [pa for pa in Loops.__KEYS if pa not in matched_parameters]
        miss.sort()
        if not miss:
            self.param = matched_parameters
        else:
            message = "{} must have {} keys".format(
                parameters.__repr__(), ','.join(miss))
            raise KeyError(message)

        Np = int(self.param["Np"])
        self.param["Temperature"] = self.Temperature

        self.Cooper = np.zeros((Np, Np, Np), float)
        self.Peierls = np.zeros((Np, Np, Np), float)
        self.Peierls_susc = np.zeros((Np, 2), float)

    def initialize(self, **kwargs: dict) -> object:
        '''
            Set Loops values to null.

        '''
        if kwargs is not None:
            if "Temperature" in kwargs:
                self.Temperature = kwargs["Temperature"]
            else:
                self.Temperature = Loops.Temperature

            for key, value in kwargs.items():
                if key in self.param:
                    self.param[key] = value
        self.param["Temperature"] = self.Temperature
        self.Cooper = np.zeros(self.Cooper.shape, float)
        self.Peierls = np.zeros(self.Peierls.shape, float)
        self.Peierls_susc = np.zeros(self.Peierls_susc.shape, float)
        return self

    def get_values(self, l_rg: float):

        # cb.loops_integration(l_rg, self.Temperature, self.param, self.Cooper,
        #                      self.Peierls, self.Peierls_susc)
        self.param["lrg"] = l_rg
        cb.loops_integration(
            l_rg, self.param, self.Cooper, self.Peierls, self.Peierls_susc
        )
        del self.param["lrg"]

    def __call__(self, l_rg: float):
        '''
            Calculate the values of Cooper and Peierls loops
            for a given Temperature and RG_l parameter.
            This is donne via cbulle module using cython
            to do integration for time considerations.

        '''
        try:
            self.param["lrg"] = l_rg
            cb.loops_integration(
                self.param, self.Cooper,
                self.Peierls, self.Peierls_susc
            )
            del self.param["lrg"]
        except Exception as identifier:
            print(identifier)
            return False
        else:
            return True

        return True
