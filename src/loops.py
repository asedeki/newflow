
import difflib

import numpy as np

try:
    import LoopsIntegration as cb
    from utils import best_match_dict
except Exception:
    import os
    import sys
    path = os.getcwd().split("/")
    path = "/".join(path[:path.index("newflow")+1])
    sys.path.append(path)
    import lib.LoopsIntegration as cb
    from src.utils import best_match_dict


# import pstats
# import cProfile


class Loops():
    __KEYS = set(["Np", "tp2", "tp", "Ef"])
    __KEYS_INT = set(["Temperature", "lrg"])
    Temperature = 1e-80
    parameters = {k: None for k in __KEYS}
    parameters.update({k: None for k in __KEYS_INT})

    def __init__(self, parameters: dict) -> None:
        try:
            Temp = difflib.get_close_matches("T", parameters, n=1)[0]
            self.Temperature = float(parameters[Temp])
        except Exception:
            pass

        matched_parameters = best_match_dict(
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
                del kwargs["Temperature"]
            else:
                self.Temperature = Loops.Temperature
            if kwargs:
                dic = best_match_dict(
                    kwargs, Loops.__KEYS
                )
                # input(f"dic = {dic}")
                self.param.update(dic)
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
            self.param, self.Cooper, self.Peierls, self.Peierls_susc
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
