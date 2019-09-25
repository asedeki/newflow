import difflib
import os
import sys
from types import MethodType

import numpy as np

# from numpy import cos, sin
path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow")+1])
else:
    path = "/".join(path.append("newflow"))
try:
    from interaction import Interaction
    from integrable import Integrable
    from loops import Loops
except ModuleNotFoundError:
    sys.path.append(path)
    from src.interaction import Interaction
    from src.integrable import Integrable
    from src.loops import Loops


def rg_equations_cbdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)

    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(2):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]

            z[0, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp]
                              - couplage.g2[i, kpp, kpp]
                              - 2*couplage.g3[i, kpp, kpp]
                              + couplage.g3[i, kpp, i]) * self.vertex[0, i]*Ipc

            z[1, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp_kp]
                              - couplage.g2[i, kpp, kpp_kp]
                              - 2*couplage.g3[i, kpp, kpp_kp]
                              + couplage.g3[i, kpp, i_kp])*self.vertex[1, i]*Ipp

    return self.pack(z, chi)


def rg_equations_csdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1

    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):

            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]

            z[0, kpp] -= 0.5*(
                2*couplage.g1[i, kpp, kpp]
                - couplage.g2[i, kpp, kpp]
                + 2*couplage.g3[i, kpp, kpp]
                - couplage.g3[i, kpp, i]
            )*self.vertex[0, i]*Ipc

            z[1, kpp] -= 0.5*(
                2 * couplage.g1[i, kpp, kpp_kp]
                - couplage.g2[i, kpp, kpp_kp]
                + 2*couplage.g3[i, kpp, kpp_kp]
                - couplage.g3[i, kpp, i_kp]
            )*self.vertex[1, i]*Ipp

    return self.pack(z, chi)


def rg_equations_sbdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf
    qperp = [0, 1]
    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]
            z[0, kpp] += 0.5*(couplage.g2[i, kpp, kpp] -
                              couplage.g3[i, kpp, i])*self.vertex[0, i]*Ipc
            z[1, kpp] += 0.5*(couplage.g2[i, kpp, kpp_kp] -
                              couplage.g3[i, kpp, i_kp])*self.vertex[1, i]*Ipp

    return self.pack(z, chi)


def rg_equations_ssdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]
            z[0, kpp] += 0.5*(couplage.g2[i, kpp, kpp] +
                              couplage.g3[i, kpp, i])*self.vertex[0, i]*Ipc
            z[1, kpp] += 0.5*(couplage.g2[i, kpp, kpp_kp] +
                              couplage.g3[i, kpp, i_kp])*self.vertex[1, i]*Ipp

    return self.pack(z, chi)


def rg_equations_supra_singlet(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)

    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Cooper[0, :, 0])

    # kp = -i_inf
    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            mkpp = (-kpp) % Np
            Ic = loops.Cooper[kpp, i, kc]
            for j in range(self.dim1):
                z[j, kpp] -= 0.5*(couplage.g1[kpp, mkpp, i]
                                  + couplage.g2[kpp, mkpp, i]
                                  )*self.vertex[j, i]*Ic
    return self.pack(z, chi)


def rg_equations_supra_triplet(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = -i_inf

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Cooper[0, :, 0])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            mkpp = (-kpp) % Np
            Ic = loops.Cooper[kpp, i, kc]
            for j in range(self.dim1):
                z[j, kpp] += 0.5*(couplage.g1[kpp, mkpp, i]
                                  - couplage.g2[kpp, mkpp, i])*self.vertex[j, i]*Ic

    return self.pack(z, chi)


class Susceptibility(Integrable):
    __SUSCEPTIBILITY_TYPE = {
        "csdw": {"dim1": 2,
                 "func_ini": None,
                 "rg": rg_equations_csdw,
                 "type": ["Site_Charge"]},
        "cbdw": {"dim1": 2,
                 "func_ini": None,
                 "rg": rg_equations_cbdw,
                 "type": ["Bond_Charge"]},
        "ssdw": {"dim1": 2,
                 "func_ini": None,
                 "rg": rg_equations_ssdw,
                 "type": ["Site_Spin"]},
        "sbdw": {"dim1": 2,
                 "func_ini": None,
                 "rg": rg_equations_sbdw,
                 "type": ["Bond_Spin"]},
        "supra_triplet": {"dim1": 4,
                          "func_ini": ["", "1*np.sin",
                                       "2*np.cos", "1*np.cos"],
                          "rg": rg_equations_supra_triplet,
                          "type": ["px", "py", "h", "f"]},
        "supra_singlet": {"dim1": 5,
                          "func_ini": ["", "1*np.sin", "1*np.cos",
                                       "2*np.sin", "3*np.cos"],
                          "rg": rg_equations_supra_singlet,
                          "type": ["s", "dxy", "dx2y2", "g", "i"]}
    }
    dim2 = None
    Neq = None
    dim1 = None
    types = None
    name = None
    susceptibilities = __SUSCEPTIBILITY_TYPE.keys()

    def __init__(self, parameters: dict):
        self.dim2 = parameters["Np"]

    def get_susceptibility_by_name(self, name: str):
        near_name = difflib.get_close_matches(
            name, list(self.__SUSCEPTIBILITY_TYPE.keys()), 1, 0.7
        )[0]
        if near_name in self.__SUSCEPTIBILITY_TYPE:
            susceptibility_type = self.__SUSCEPTIBILITY_TYPE[near_name]
            self.dim1 = susceptibility_type["dim1"]
            self.Neq = self.dim1*(self.dim2 + 1)
            self.initialize(func_ini=susceptibility_type["func_ini"])
            self.rg_equations = MethodType(susceptibility_type["rg"], self)
            self.types = susceptibility_type["type"]
            self.name = near_name
        else:
            raise KeyError(f"{name} is not a valid susceptibility name")

    def pack(self, d_vertex: np.ndarray, d_susc: np.ndarray):
        y = np.zeros(self.Neq, float)
        y[:self.dim1] = d_susc
        y[self.dim1:] = d_vertex.reshape(self.dim2*self.dim1)
        return y

    def initpack(self) -> np.ndarray:
        y = np.zeros(self.Neq, float)
        y[:self.dim1] = self.susceptibility
        y[self.dim1:] = self.vertex.reshape(self.dim2*self.dim1)
        return y

    def unpack(self, y: np.ndarray):
        self.susceptibility = y[:self.dim1]
        self.vertex = y[self.dim1:].reshape(self.dim1, self.dim2)

    def initialize(self, func_ini):
        self.vertex = np.zeros((self.dim1, self.dim2), float)
        self.susceptibility = np.zeros(self.dim2, float)
        v = 2*np.pi/float(self.dim2)
        k_perp = np.arange(self.dim2) * v
        self.vertex[:, :] = 1.0
        if func_ini is not None:
            for i in range(1, self.dim1):
                list_function = func_ini[i].split("*")
                CONSTANTE = float(list_function[0])
                function = list_function[1]
                self.vertex[i, :] = np.sqrt(
                    2) * eval(function)(CONSTANTE * k_perp)

    def rg_equations(self, l_rg: float):
        # return self.derivative
        pass


class Susceptibilities(Integrable):
    def __init__(self, parameters: dict, susceptibilities_types: list):
        self.susceptibilities = {}
        self.parameters = parameters
        self.susceptibilities_types = susceptibilities_types
        self.susceptibilities_types.sort()
        self.initialize()
        self.Neq = self.__get_variables_number()
    
    def __get_variables_number(self):
        Neq = 0
        for susc in self.susceptibilities.values():
            Neq += susc.Neq
        return Neq

    def initialize(self):
        for name in self.susceptibilities_types:
            susc = Susceptibility(self.parameters)
            susc.get_susceptibility_by_name(name=name)
            self.susceptibilities[name] = susc

    def derivative(self, loops: Loops, interaction: Interaction):
        derivs = {}
        for name, susc in self.susceptibilities.items():
            derivs[name] = susc.rg_equations(loops, interaction)
        return self.pack(derivs)

    def pack(self, derivs: dict):
        y = np.zeros(self.Neq, float)
        indice = 0
        for name in self.susceptibilities_types:
            neq = self.susceptibilities[name].Neq
            y[indice:indice + neq] = derivs[name]
            indice += neq
        return y

    def unpack(self, y: np.ndarray):
        indice = 0
        for name in self.susceptibilities_types:
            neq = self.susceptibilities[name].Neq
            self.susceptibilities[name].unpack(y[indice:indice+neq])
            indice += neq


if __name__ == "__main__":
    r = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 8,
         "g1": 0.2, "g2": 0.2, "g3": 0.2}
    g = Interaction(r)

    rf = Susceptibility(r)
    b = Loops(r)
    b.resset()
    b(1, 2)
    rf.get_susceptibility_by_name("csdw")
    y = rf.rg_equations(b, g)
    print(y)

    for ty in rf.susceptibilities:
        rf2 = Susceptibility(r)
        print(ty)
        rf2.get_susceptibility_by_name(ty)
        y = rf2.rg_equations(b, g)
        print(y)
