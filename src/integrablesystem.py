import os
import sys
import numpy as np

path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow")+1])
else:
    path.append("newflow")
    path = "/".join(path)
sys.path.append(path)

from src.integrable import Integrable
from src.interaction import Interaction
from src.quasi1dsusceptibilities import Susceptibilities
from src.loops import Loops


class IntegrableQuasi1dSystem(Integrable):
    def __init__(self):
        self.interaction = None
        self.susceptibilities = None
        self.loops = None
        self.Neq = None

    def set_all(self, parameters: dict, susc_name: str = None):
        self.parameters = parameters
        self.interaction = Interaction(parameters)
        self.susceptibilities = Susceptibilities().append_all(susc_name)
        self.loops = Loops(**parameters)
        self.Neq = self.interaction.Neq
        for s in self.susceptibilities.susceptibilities:
            self.Neq += s.dim1*(parameters["Np"]+1)
        return self

    def set_interaction(self, g: Interaction):
        if not (isinstance(g, Interaction)):
            raise TypeError(
                f"{g} must be a <class 'Interaction'>"
            )
        self.interaction = g

    def set_susceptibilities(self, susceptibilities: Susceptibilities):
        if not (isinstance(susceptibilities, Susceptibilities)):
            raise TypeError(
                f"{susceptibilities} must be a <class Susceptibilities>."
            )
        self.susceptibilities = susceptibilities

    def set_loops(self, loops: Loops):
        if not (isinstance(loops, Loops)):
            raise TypeError(
                f"{loops} must be a <class Loops> object."
            )
        self.loops = loops

    def initialize(self, **kwargs):
        self.loops.initialize(**kwargs)
        self.interaction.initialize(**kwargs)
        self.susceptibilities.initialize(Np=self.parameters["Np"])

    def initpack(self) -> np.ndarray:
        y = np.concatenate(
            (
                self.interaction.initpack(),
                self.susceptibilities.initpack()
            )
        )
        return y

    def pack(self, *vargs) -> np.ndarray:
        pass

    def unpack(self, y: np.ndarray) -> None:
        self.interaction.unpack(y[:self.interaction.Neq])
        self.susceptibilities.unpack(y[self.interaction.Neq:])

    def rg_equations(self, lflow: float) -> np.ndarray:
        self.loops(lflow=lflow)
        # input(f"T_loops={self.loops.Temperature}")
        dy = self.interaction.rg_equations(loops=self.loops)
        dy = np.concatenate(
            (dy, self.susceptibilities.rg_equations(
                self.loops, self.interaction))
        )
        return dy

    def __iter__(self):
        return self

    def __next(self):
        pass


class IntegrableSystem(Integrable):
    def __init__(self):
        self.integrable_systems = []
        self.size = 0

    def add_integrable_system(self, *integ_systems: Integrable):
        for integ_sys in integ_systems:
            if not (isinstance(integ_sys, Integrable)):
                raise TypeError(
                    f"{integ_sys} must be a <class 'Integrable'> object."
                )
            else:
                self.integrable_systems.append(integ_sys)
                self.size += 1

    def remove_integrable_system(self, *integ_systems: Integrable):
        assert(self.size > 0)
        for integ_sys in integ_systems:
            if integ_sys not in self.integrable_systems:
                raise ValueError(
                    f"{integ_sys} not found"
                )
            self.integrable_systems.remove(integ_sys)
            self.size -= 1

    def remove_all(self):
        self.integrable_systems.clear()
        self.size = 0

    def initialize(self, **kwargs):
        for integ_sys in self.integrable_systems:
            integ_sys.initialize(**kwargs)

    def initpack(self, *vargs) -> np.ndarray:
        y = self.integrable_systems[0].initpack()
        for i in range(1, self.size):
            y = np.concatenate(
                (y, self.integrable_systems[i].initpack())
            )
        return y

    def pack(self, *vargs) -> np.ndarray:
        pass

    def unpack(self, y: np.ndarray) -> None:
        index_ini = 0
        index_final = self.integrable_systems[0].Neq
        self.integrable_systems[0].unpack(y[index_ini:index_final])
        index_ini = index_final
        for i in range(1, self.size):
            integ_sys = self.integrable_systems[i]
            index_final += integ_sys.Neq
            integ_sys.unpack(y[index_ini:index_final])
            index_ini += integ_sys.Neq

    def rg_equations(self, lflow: float) -> np.ndarray:
        dy = self.integrable_systems[0].rg_equations(lflow=lflow)
        for i in range(1, self.size):
            integ_sys = self.integrable_systems[i]
            dy = np.concatenate(
                (dy, integ_sys.rg_equations(lflow=lflow))
            )
        return dy

    def __iter__(self):
        return self

    def __next(self):
        pass


if __name__ == "__main__":
    parameters = {
        "tp": 200, "tp2": 20,
        "Ef": 3000, "Np": 4,
        "g1": 0, "g2": 0, "g3": 0
    }
    d = IntegrableQuasi1dSystem(parameters=parameters)
    d.set_all()
    print(
        list(d.__dict__.keys())
    )
