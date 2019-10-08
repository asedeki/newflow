import os
import sys
import warnings

import numpy as np
import scipy.integrate as scipyInt

path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path.append("newflow")
    path = "/".join(path)

sys.path.append(path)
from src.integrable import Integrable
from src.integrablesystem import IntegrableSystem


class Dynamic():
    def __init__(self, rel_tol: float = 1e-3):
        """
            ;param ode_method in ['dop853','dopri5','lsoda']
            default = 'dop853'
        """
        self.dynamical_sytem = None
        self.ode_integrator = None
        self.rtol = rel_tol

    def initialize(self, **kwargs):
        """ Description
        :type self:
        :param self:

        :type **kwargs:
        :param **kwargs:

        :raises:

        :rtype:
        """

        self.dynamical_sytem.initialize(**kwargs)

    def get_integrator(self, dynamical_system: Integrable = None, ode_method: str = 'dop853'):
        self.ode_method = ode_method
        if dynamical_system is not None:
            self.set_dynamical_system(dynamical_system)
        self.__get_integrator()
        return self

    def __get_init_value(self) -> np.ndarray:
        y = self.dynamical_sytem.initpack()
        return y

    def __get_integrator(self):
        def rg(l, y):
            dy = self.__derivative(y=y, lflow=l)
            return dy
        self.ode_integrator = scipyInt.ode(rg).set_integrator(
            self.ode_method, rtol=self.rtol)
        # self.ode_integrator.set_f_params(9)

    def set_dynamical_system(self,
                             dynamical_system: Integrable) -> None:

        if not (isinstance(dynamical_system, Integrable)):
            print(f"{dynamical_system} must be a <class 'Integrable'> object.")
            raise TypeError
        (
            f"{dynamical_system} must be a <class 'Integrable'> object."
        )
        self.dynamical_sytem = dynamical_system

    def __derivative(self, y: np.ndarray,
                     lflow: float) -> np.ndarray:
        self.dynamical_sytem.unpack(y)
        dy = self.dynamical_sytem.rg_equations(lflow=lflow)

        return dy

    def next_value(self, l_ini: float = 0.0,
                   l_next: float = 100.0,
                   **kwargs: dict) -> bool:

        y0 = self.__get_init_value()
        self.ode_integrator.set_initial_value(y0, l_ini)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.ode_integrator.integrate(l_next)
        if (self.ode_integrator.successful()):
            self.dynamical_sytem.unpack(self.ode_integrator.y)

        return self.ode_integrator.successful()

    def next(self, l_ini: float = 0.0,
             l_next: float = 100.0,
             **kwargs) -> bool:

        def rg(y, l):
            dy = self.__derivative(y=y, lflow=l)
            return dy

        y0 = self.__get_init_value()
        y = scipyInt.odeint(rg, y0, [l_ini, l_next], rtol=self.rtol)
        self.dynamical_sytem.unpack(y[1])
    # # Context manager
    # def __enter__(self):
    #     pass

    # # Context manager
    # def __exit__(self, *vargs, **kwargs):
    #     pass


if __name__ == "__main__":
    from interaction import Interaction
    parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32,
                  "g1": 0.1, }
    c = Dynamic()

    c.set_dynamical_system(["a"])
