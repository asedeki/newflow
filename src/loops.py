
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
    def __new__(cls, *args, **kwargs):
        instance = super(Loops, cls).__new__(cls)
        instance.__init()
        return instance

    @classmethod
    def __init(cls):
        cls.__KEYS = set(["Np", "tp2", "tp", "Ef"])
        cls.__KEYS_INT = set(["Temperature", "lrg"])
        cls.Temperature = None

        cls.parameters = {k: None for k in cls.__KEYS}
        cls.parameters["lrg"] = None
        cls.parameters["Temperature"] = 1e-80
        cls.Cooper = None
        cls.Peierls = None
        cls.Peierls_susc = None

    def __init__(self, parameters: dict = None) -> None:

        if parameters is not None:
            matched_parameters = best_match_dict(
                parameters, Loops.__KEYS.union(Loops.__KEYS_INT)
            )
            miss = [pa for pa in Loops.__KEYS if pa not in matched_parameters]
            miss.sort()
            # # input(f"before = {self.parameters}")
            # # input(f"match = {matched_parameters}")
            self.parameters.update(matched_parameters)
            # # input(f"after = {self.parameters}")

            if miss:
                message = "{} must have {} keys".format(
                    parameters.__repr__(), ','.join(miss))
                raise KeyError(message)

            Np = int(self.parameters["Np"])
            self.Temperature = self.parameters["Temperature"]

    def initialize(self, **kwargs: dict) -> None:
        '''
            Set Loops values to null.

        '''

        if kwargs:
            dic = best_match_dict(
                kwargs, set(self.parameters.keys())
            )
            # # # input(f"dic = {dic}")
            self.parameters.update(dic)
        # input(f"param_in_intialize = {self.parameters}")
        self.Temperature = self.parameters["Temperature"]
        assert(self.parameters["Np"] is not None)
        shape = (self.parameters["Np"], self.parameters["Np"],
                 self.parameters["Np"])
        self.Cooper = np.zeros(shape, float)
        self.Peierls = np.zeros(shape, float)
        self.Peierls_susc = np.zeros((self.parameters["Np"], 2), float)

    def get_values(self, **kwargs: dict):

        # cb.loops_integration(l_rg, self.Temperature, self.parameters, self.Cooper,
        #                      self.Peierls, self.Peierls_susc)
        self.initialize(**kwargs)
        self.__call__()

    def __call__(self, l_rg: float = None):
        '''
            Calculate the values of Cooper and Peierls loops
            for a given Temperature and RG_l parameterseter.
            This is donne via cbulle module using cython
            to do integration for time considerations.

        '''
        if l_rg is not None:
            self.parameters["lrg"] = l_rg
        try:
            cb.loops_integration(
                self.parameters, self.Cooper,
                self.Peierls, self.Peierls_susc
            )
        except Exception as identifier:
            print(identifier)
            return False
        else:
            return True
