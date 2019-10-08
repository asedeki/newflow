'''
created on august 25, 2019
@author: Abdelouahab Sedeki
@company: Saida University (Algeria)
@email: asedeki@gmail.com
'''
# TODO Utilisation de jitclass de numba :
# TODO http://numba.pydata.org/numba-doc/latest/user/jitclass.html?highlight=jitclass
# IDEA : utilise numba jitclass
# import concurrent.futures as concfut
import sys
import os

import numpy as np

path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path.append("newflow")
    path = "/".join(path)
sys.path.append(path)

from src.integrable import Integrable
from src.loops import Loops
import src.utils as utils

# todo reflechir a wrap(), %N ne donne que des valeurs positives


class Interaction(Integrable):
    '''
        Class representing the goology modele interaction:
            g1, g2, umklapp g3
    '''
    _params = ["g1", "g2", "g3", "Np"]

    def __init__(self, **parameters):
        self.parameters = {k: None for k in Interaction._params + Loops._params}
        self.parameters.update(
            utils.best_match_dict(parameters,
                Interaction._params + Loops._params
                )
        )
        
        Np = self.parameters["Np"]

        if Np is None:
            raise ValueError('The number of patches Np must be given')
        
        self.Ng = Np**3
        self.ndim = (Np, Np, Np)
        self.Neq = 3 * self.Ng
        self.loops = Loops(**parameters)

    def initialize(self, **kwargs):
        self.parameters.update(
            utils.best_match_dict(kwargs, self._params)
        )

        self.g1 = np.ones(self.ndim, float) * float(self.parameters["g1"])
        self.g2 = np.ones(self.ndim, float) * float(self.parameters["g2"])
        self.g3 = np.ones(self.ndim, float) * float(self.parameters["g3"])
        self.loops.initialize(**kwargs)

    def initpack(self):
        '''
            Summary line
            Extended description of the function

            Parameters:
                self: Interaction

            return:
                numpy array: representing concactenation
                of the interaction arrays (g1,g2,g3).
        '''
        y = np.concatenate((
            self.g1.reshape((self.Ng,)),
            self.g2.reshape((self.Ng,)),
            self.g3.reshape((self.Ng,))
        ))
        return y

    def pack(self, dg1, dg2, dg3):
        '''
            Summary line
            Extended description of the function

            Parameters:
            ---------------------------------------
                dg1: float numpy array
                dg2: float numpy array
                dg3: float numpy array
            return:
            ----------------------------------------
                numpy array
                    representing concactenation of the arrays (dg1,dg2,dg3)
        '''
        y = np.concatenate((
            dg1.reshape((self.Ng,)),
            dg2.reshape((self.Ng,)),
            dg3.reshape((self.Ng,))
        ))
        del dg1
        del dg2
        del dg3

        return y

    def unpack(self, y: np.ndarray):
        '''
            Summary line
            Extended description of the function

            Parameters:
            ---------------------------------------
                y: float numpy array

            return:
            ----------------------------------------
                None
                    execute the spliting of the y array (# # input)
                    in three arrays self.g1, self.g2, self.g3
        '''
        self.g1 = y[:self.Ng].reshape(self.ndim)
        self.g2 = y[self.Ng:2 * self.Ng].reshape(self.ndim)
        # self.g3 = y[2 * self.Ng:3 * self.Ng].reshape(self.ndim)
        self.g3 = y[2 * self.Ng:].reshape(self.ndim)
    def rg_equations(self, lflow: float):
        """[summary]

        Keyword Arguments:
            loops {Loops} -- [description] (default: {None})
            lflow {float} -- [description] (default: {0})

        Returns:
            [ndarray] -- [Contient les derivees de g1, g2, g3]

            It's a pure python function calculating
            the RG derivatives of the goology modele
            interactions. It's uses "numba.jil" to
            speed up the loops execution!!!
            Parameters:
            ---------------------------------------
                loops: Loops class
                    contains the values of the Cooper
                    and Peierls loops values.

            return:
            ----------------------------------------
                numpy array

                    Calculate the rg derivatives for the g1, g2, g3
                    goology interaction. The result is concatenated
                    in an numpy array via the pack function.
        """
        self.loops(lflow=lflow)
        dg1 = np.zeros(self.ndim, float)
        dg2 = np.zeros(self.ndim, float)
        dg3 = np.zeros(self.ndim, float)
        utils.rg_equations_interaction(dg1, dg2, dg3,
                                       self.g1, self.g2, self.g3,
                                       self.loops.Peierls, self.loops.Cooper)
        return self.pack(dg1, dg2, dg3)
