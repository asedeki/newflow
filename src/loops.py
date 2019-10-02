
import difflib
import numpy as np
import os
import sys
path = os.getcwd().split("/")

if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path.append("newflow")
    path = "/".join(path)
sys.path.append(path)

try:
    import lib.LoopsIntegration as cb
    import src.utils as utils
except ImportError as e:
    print(f"path={path}")
    raise ImportError

# import pstats
# import cProfile
# TODO: rajouter jil a la class Dynamical


class Loops():
    _params = ["Np", "tp2", "tp", "Ef",
               "Temperature", "lflow"]

    # def __new__(cls, *args, **kwargs):
    #     instance = super(Loops, cls).__new__(cls)
    #     return instance
    Cooper = None
    Peierls = None
    Peierls_susc = None
    loops_donne = False

    def __init__(self, **parameters: dict) -> None:
        # self.Cooper = None
        # self.Peierls = None
        # self.Peierls_susc = None
        # self.loops_donne = False
        self.parameters = {k: None for k in self._params}
        if parameters is not None:
            self.parameters.update(
                utils.best_match_dict(parameters, self._params)
            )

    def initialize(self, **kwargs: dict) -> None:
        '''
            Set Loops values to null.

        '''

        if kwargs:
            self.parameters.update(
                utils.best_match_dict(kwargs, self._params)
            )
        # input(f"param_in_intialize = {self.parameters}")
        # print(f"lflow_ini = {self.parameters['lflow']}")
        # self._assert_parameters_not_none()

        shape = (self.parameters["Np"], self.parameters["Np"],
                 self.parameters["Np"])
        self.Cooper = np.zeros(shape, float)
        self.Peierls = np.zeros(shape, float)
        self.Peierls_susc = np.zeros((self.parameters["Np"], 2), float)

    def _assert_parameters_not_none(self):
        _none_param = []
        for p in self.parameters:
            if self.parameters[p] is None and p != "lflow":
                _none_param.append(p)
        if _none_param:
            raise ValueError(
                f"Loop's ({','.join(_none_param)}) parameters must be given."
            )

    def get_values(self, **kwargs):

        # cb.loops_integration(l_rg, self.Temperature, self.parameters, self.Cooper,
        #                      self.Peierls, self.Peierls_susc)
        self.initialize(**kwargs)
        self.__call__()

    def __call__(self, **kwargs):
        '''
            Calculate the values of Cooper and Peierls loops
            for a given Temperature and RG_l parameterseter.
            This is donne via cbulle module using cython
            to do integration for time considerations.

        '''
        if kwargs:
            self.parameters.update(
                utils.best_match_dict(kwargs, self._params)
            )

        # print(f"lflow_call = {self.parameters['lflow']}")
        self._assert_parameters_not_none()
        try:
            Loops.loops_donne = cb.loops_integration(
                self.parameters, self.Cooper,
                self.Peierls, self.Peierls_susc
            )
        except Exception as identifier:
            print(identifier)
            return False
        else:
            return True

    def __delete__(self, name):
        self.parameters = {}
        input(Loops.Cooper)
        return super().__delattr__(self)
