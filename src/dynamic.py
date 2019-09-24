import os
import sys
import warnings

import numpy as np
from scipy.integrate import ode, solve_ivp

path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow")+1])
else:
    path = "/".join(path.append("newflow"))
try:
    from integrable import Integrable
    from dynamicalsystem import DynamicalSystem

except Exception:
    sys.path.append(path)
    from src.integrable import Integrable
    from src.dynamicalsystem import DynamicalSystem


class Dynamic():
    dynamical_sytem = None
    rtol = None
    ode_method = None
    ode_integrator = None

    def __init__(self, rel_tol: float = 1e-3,
                 ode_method: str = 'dop853'):
        """
            ;param ode_method in ['dop853','dopri5','lsoda']
            default = 'dop853'
        """
        self.rtol = rel_tol
        self.ode_method = ode_method

    def get_integrator(self, dynamical_system: DynamicalSystem,
                       **kwargs):
        self.set_dynamical_system(dynamical_system)
        self.initialize(**kwargs)
        self.__get_integrator()
        return self

    def __get_integrator(self):
        def rg(l, y):
            dy = self.__derivative(y=y, lflow=l)
            return dy
        self.ode_integrator = ode(rg).set_integrator(
            self.ode_method, rtol=self.rtol)

    def set_dynamical_system(self, dynamical_sytem: DynamicalSystem):
        # for system in dynamical_sytem:
        #     if not (isinstance(system, Integrable)):
        #         raise TypeError(
        #             f"{system} must be a <class 'Integrable'> object."
        #         )
        self.dynamical_sytem = dynamical_sytem

    def initialize(self, **kwargs):
        self.dynamical_sytem.initialize(**kwargs)

    def __get_init_value(self) -> np.ndarray:
        y = self.dynamical_sytem.initpack()
        return y

    def unpack(self, y: np.array) -> None:
        self.dynamical_sytem.unpack(y)

    def __derivative(self, y: np.ndarray, lflow: float) -> np.ndarray:
        dy = self.dynamical_sytem.rg_equations(y=y, lflow=lflow)
        return dy

    def next_value(self, l_ini: float = 0.0,
                   l_next: float = 100.0,
                   **kwargs: dict) -> bool:
        # Avoir si self.ode_integrator marche ou pas
        # def rg(l, y):
        #     dy = self.__derivative(y, l)
        #     return dy
        # ode_integrator = ode(rg).set_integrator(
        #     self.ode_method, rtol=self.rtol)

        # ode_integrator.set_f_params(9)
        y0 = self.__get_init_value()
        self.ode_integrator.set_initial_value(y0, l_ini)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.ode_integrator.integrate(l_next)
        if (self.ode_integrator.successful()):
            self.unpack(self.ode_integrator.y)
        return self.ode_integrator.successful()

    def evolutionl(self, li, lf):
        def rg(l, y):
            dy = self.__derivative(y, l)
            return dy
        l_rg = [li, lf]
        y0 = self.__get_init_value()
        with warnings.catch_warnings():
            sol = solve_ivp(rg, l_rg, y0, t_eval=[li, lf])
            if (sol.success):
                self.unpack(sol.y[:, -1])
                return True
            else:
                # print(sol.message)
                return False

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