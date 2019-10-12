import os
import sys
import numpy as np

path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path.append("newflow")
    path = "/".join(path)
sys.path.append(path)
from src.loops import Loops
from src.quasi1dsusceptibilities import Susceptibilities
from src.interaction import Interaction
from src.integrable import Integrable


class Quasi1dIntegrableSystem(Integrable):
    def __init__(self):
        self.interaction = None
        self.susceptibilities = None

    def set_all(self, susc_name: str = None, **parameters):
        Np = parameters["Np"]
        self.interaction = Interaction(**parameters)
        if susc_name is None:
            self.susceptibilities = Susceptibilities(Np=Np).append_all()
        else:
            self.susceptibilities = Susceptibilities(Np=Np)
            self.susceptibilities.append_susceptibility_by_name(
                name_string=susc_name
            )
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

    def initialize(self, **kwargs):
        self.interaction.initialize(**kwargs)
        self.susceptibilities.initialize()

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
        dy = np.concatenate(
            (
                self.interaction.rg_equations(lflow=lflow),
                self.susceptibilities.rg_equations(self.interaction)
            )
        )
        return dy

    # def __iter__(self):
    #     return self

    # def __next(self):
    #     pass


class IntegrableSystem(Integrable):
    def __init__(self):
        self.integrable_systems = list()
        self.size = 0
        self.Neq = 0

    def add_integrable_system(self, *integ_systems: Integrable):
        for integ_sys in integ_systems:
            if not (isinstance(integ_sys, Integrable)):
                raise TypeError(
                    f"{integ_sys} must be a <class 'Integrable'> object."
                )
            else:
                self.integrable_systems.append(integ_sys)
                self.size += 1
                self.Neq += integ_sys.Neq

    def remove_integrable_system(self, *integ_systems: Integrable):
        assert(self.size > 0)
        for integ_sys in integ_systems:
            if integ_sys not in self.integrable_systems:
                raise ValueError(
                    f"{integ_sys} not found"
                )
            self.integrable_systems.remove(integ_sys)
            self.size -= 1
            self.Neq -= integ_sys.Neq

    def remove_all(self):
        self.integrable_systems.clear()
        self.size = 0
        self.Neq = 0

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
        index_final = 0
        for i in range(self.size):
            index_final += self.integrable_systems[i].Neq
            self.integrable_systems[i].unpack(y[index_ini:index_final])
            index_ini = index_final

    def rg_equations(self, lflow: float) -> np.ndarray:
        dy = self.integrable_systems[0].rg_equations(lflow=lflow)
        for i in range(1, self.size):
            dy = np.concatenate(
                (dy,
                 self.integrable_systems[i].rg_equations(lflow=lflow)
                 )
            )
        return dy

    # def __iter__(self):
    #     return self

    # def __next(self):
    #     pass


if __name__ == "__main__":
    parameters = {
        "tp": 0, "tp2": 0,
        "Ef": 3000, "Np": 2,
        "g1": 0.2, "g2": 0.64, "g3": 0.03,
        "Temperature": 100
    }
    S = Susceptibilities(Np=2).append_all()
    S.delete_susceptibility_by_name('csdw cbdw singhlet triplet')
    g = Interaction(**parameters)

    # d = Quasi1dIntegrableSystem().set_all(**parameters)
    d = Quasi1dIntegrableSystem()
    d.set_interaction(g)
    d.set_susceptibilities(S)
    d.initialize(Temperature=1e-80)
    y = d.rg_equations(lflow=1)
    print(d.interaction.loops.Cooper)
    d.unpack(y)
    for name, _, _ in d.susceptibilities:
        print(name)
