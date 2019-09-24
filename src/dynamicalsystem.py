import os
import sys
import numpy as np

path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow")+1])
else:
    path = "/".join(path.append("newflow"))
try:
    from integrable import Integrable
    from interaction import Interaction
    from susceptibilities import Susceptibility
    from loops import Loops
except ModuleNotFoundError:
    sys.path.append(path)
    from src.integrable import Integrable
    from src.interaction import Interaction
    from src.susceptibilities import Susceptibility
    from src.loops import Loops


class DynamicalSystem(Integrable):
    # interaction = None
    # susceptibilities = {}
    # susc_types = None
    # loops = None

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.interaction = None
        self.susceptibilities = {}
        self.susc_types = None
        self.loops = None

    def set_all(self):
        self.set_interaction()
        self.set_suscptibilities_by_name(
            " ".join(list(
                Susceptibility._Susceptibility__SUSCEPTIBILITY_TYPE.keys()
            )
            )
        )
        return self

    def resset_parameters(self, **kwargs):
        for key, v in kwargs.items():
            if hasattr(self.parameters, key):
                self.parameters[key] = v

    def set_interaction(self, g: Interaction = None):
        if g is not None:
            if not (isinstance(g, Integrable)):
                raise TypeError(
                    f"{g} must be a <class 'Integrable'> object."
                )
            self.interaction = g
        else:
            self.interaction = Interaction(self.parameters)

    def set_suscptibilities_by_name(self, type_name: str):
        if isinstance(type_name, str):
            susc_types = type_name.split(" ")
            # input(f"susc = {susc_types}")
            for name in susc_types:
                try:
                    s = Susceptibility(self.parameters)
                    s.set_susceptibility_by_name(name)
                    self.susceptibilities[name] = s
                except Exception as e:
                    print(e)

        self.susc_types = list(self.susceptibilities.keys())
        self.susc_types.sort()

    def set_suscptibilities(self, susceptibilities: dict):
        if isinstance(susceptibilities, dict):
            self.susceptibilities = susceptibilities

        self.susc_types = list(self.susceptibilities.keys())
        self.susc_types.sort()

    def set_loops(self, loops: Loops = None):
        if loops is not None:
            self.loops = loops
        else:
            self.loops = Loops(self.parameters)

    def initialize(self, **kwargs):
        self.loops.initialize(**kwargs)
        self.interaction.initialize(**kwargs)
        # input(self.susceptibilities.values())
        for susc in self.susceptibilities.values():
            susc.initialize(**kwargs)

    def initpack(self, *vargs) -> np.ndarray:
        y = self.interaction.initpack()
        for susc_type in self.susc_types:
            susc = self.susceptibilities[susc_type]
            y = np.concatenate((y, susc.initpack()))
        return y

    def pack(self, *vargs) -> np.ndarray:
        pass

    def unpack(self, y: np.ndarray) -> None:
        index_ini = 0
        index_final = self.interaction.Neq
        self.interaction.unpack(y[index_ini:index_final])
        index_ini = index_final
        for susc_type in self.susc_types:
            susc = self.susceptibilities[susc_type]
            index_final += susc.Neq
            susc.unpack(y[index_ini:index_final])
            index_ini += susc.Neq

    def rg_equations(self, y: np.ndarray, lflow: float) -> np.ndarray:
        self.unpack(y)
        self.loops(l_rg=lflow)
        dy = self.interaction.rg_equations()
        for susc_type in self.susc_types:
            susc = self.susceptibilities[susc_type]
            dy = np.concatenate(
                (dy, susc.rg_equations())
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
    d = DynamicalSystem(parameters)
    d.set_all()
    print(
        list(d.__dict__.keys())
    )
