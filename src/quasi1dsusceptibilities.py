import difflib
import os
import sys
from types import MethodType
import numpy as np
from numba import njit, jit

# from numpy import cos, sin
path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path.append("newflow")
    path = "/".join(path)
sys.path.append(path)

from src.interaction import Interaction
from src.integrable import Integrable
from src.loops import Loops


def rg_equations_cbdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)

    Np = loops.Peierls.shape[0]
    i_sup = Np // 2
    i_inf = -i_sup
    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp - kp) % Np
            i_kp = (i - kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]

            z[0, kpp] -= 0.5 * (2 * couplage.g1[i, kpp, kpp]
                                - couplage.g2[i, kpp, kpp]
                                - 2 * couplage.g3[i, kpp, kpp]
                                + couplage.g3[i, kpp, i]) * self.vertex[0, i] * Ipc

            z[1, kpp] -= 0.5 * (2 * couplage.g1[i, kpp, kpp_kp]
                                - couplage.g2[i, kpp, kpp_kp]
                                - 2 * couplage.g3[i, kpp, kpp_kp]
                                + couplage.g3[i, kpp, i_kp]) * self.vertex[1, i] * Ipp

    return self.pack(z, chi)


def rg_equations_csdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np // 2
    i_inf = -i_sup

    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):

            kpp_kp = (kpp - kp) % Np
            i_kp = (i - kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]

            z[0, kpp] -= 0.5 * (
                2 * couplage.g1[i, kpp, kpp]
                - couplage.g2[i, kpp, kpp]
                + 2 * couplage.g3[i, kpp, kpp]
                - couplage.g3[i, kpp, i]
            ) * self.vertex[0, i] * Ipc

            z[1, kpp] -= 0.5 * (
                2 * couplage.g1[i, kpp, kpp_kp]
                - couplage.g2[i, kpp, kpp_kp]
                + 2 * couplage.g3[i, kpp, kpp_kp]
                - couplage.g3[i, kpp, i_kp]
            ) * self.vertex[1, i] * Ipp

    return self.pack(z, chi)


def rg_equations_sbdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np // 2
    i_inf = -i_sup
    kc = 0
    kp = i_inf
    qperp = [0, 1]
    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp - kp) % Np
            i_kp = (i - kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]
            z[0, kpp] += 0.5 * (couplage.g2[i, kpp, kpp] -
                                couplage.g3[i, kpp, i]) * self.vertex[0, i] * Ipc
            z[1, kpp] += 0.5 * (couplage.g2[i, kpp, kpp_kp] -
                                couplage.g3[i, kpp, i_kp]) * self.vertex[1, i] * Ipp

    return self.pack(z, chi)


def rg_equations_ssdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np // 2
    i_inf = -i_sup
    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp - kp) % Np
            i_kp = (i - kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]
            z[0, kpp] += 0.5 * (couplage.g2[i, kpp, kpp] +
                                couplage.g3[i, kpp, i]) * self.vertex[0, i] * Ipc
            z[1, kpp] += 0.5 * (couplage.g2[i, kpp, kpp_kp] +
                                couplage.g3[i, kpp, i_kp]) * self.vertex[1, i] * Ipp

    return self.pack(z, chi)


def rg_equations_supra_singlet(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)

    Np = loops.Peierls.shape[0]
    i_sup = Np // 2
    i_inf = -i_sup
    kc = 0
    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Cooper[0, :, 0])

    # kp = -i_inf
    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            mkpp = (-kpp) % Np
            Ic = loops.Cooper[kpp, i, kc]
            for j in range(self.dim1):
                z[j, kpp] -= 0.5 * (couplage.g1[kpp, mkpp, i]
                                    + couplage.g2[kpp, mkpp, i]
                                    ) * self.vertex[j, i] * Ic
    return self.pack(z, chi)


def rg_equations_supra_triplet(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np // 2
    i_inf = -i_sup
    kc = 0
    kp = i_inf

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Cooper[0, :, 0])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            mkpp = (-kpp) % Np
            Ic = loops.Cooper[kpp, i, kc]
            for j in range(self.dim1):
                z[j, kpp] += 0.5 * (couplage.g1[kpp, mkpp, i]
                                    - couplage.g2[kpp, mkpp, i]) * self.vertex[j, i] * Ic

    return self.pack(z, chi)


class Susceptibility(Integrable):
    SUSCEPTIBILITY_TYPE = {
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
        "supra_singulet": {"dim1": 5,
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
    SUSCEPTIBILITY_TYPE_KEYS = SUSCEPTIBILITY_TYPE.keys()
    vertex = None
    susceptibility = None
    i_dim1 = 0
    i_dim2 = 0

    def __init__(self, name: str):
        near_name = difflib.get_close_matches(
            name, list(Susceptibility.SUSCEPTIBILITY_TYPE.keys()), 1, 0.6
        )
        if near_name:
            near_name = near_name[0]
            susceptibility_type = Susceptibility.SUSCEPTIBILITY_TYPE[
                near_name]
            self.dim1 = susceptibility_type["dim1"]
            self.func_ini = susceptibility_type["func_ini"]
            self.rg_equations = MethodType(
                susceptibility_type["rg"], self
            )
            self.types = susceptibility_type["type"]
            self.name = near_name
        else:
            raise KeyError(
                f"""{name} is not a valid susceptibility name
            {Susceptibility.SUSCEPTIBILITY_TYPE.keys()}
            """)

    def pack(self, d_vertex: np.ndarray, d_susc: np.ndarray):
        # input(f"{d_vertex.shape}= 2Np")
        # input(f"{d_susc.shape} = Np")
        y = np.zeros(self.Neq, float)
        y[:self.dim1] = d_susc
        y[self.dim1:] = d_vertex.reshape(self.dim2 * self.dim1)
        return y

    def initpack(self) -> np.ndarray:
        y = np.zeros(self.Neq, float)
        y[:self.dim1] = self.susceptibility
        y[self.dim1:] = self.vertex.reshape(self.dim2 * self.dim1)
        return y

    def unpack(self, y: np.ndarray):
        self.susceptibility = y[:self.dim1]
        self.vertex = y[self.dim1:].reshape(self.dim1, self.dim2)

    def initialize(self, Np: int):
        self.dim2 = Np
        self.Neq = self.dim1 * (self.dim2 + 1)
        self.vertex = np.zeros((self.dim1, self.dim2), float)
        self.susceptibility = np.zeros(self.dim1, float)
        k_perp = 2.0 * np.arange(self.dim2) * np.pi / float(self.dim2)
        self.vertex[:, :] = 1.0
        if self.func_ini is not None:
            for i in range(1, self.dim1):
                list_function = self.func_ini[i].split("*")
                CONSTANTE = float(list_function[0])
                function = list_function[1]
                self.vertex[i, :] = np.sqrt(
                    2) * eval(function)(CONSTANTE * k_perp)

    def rg_equations(self, l_rg: float):
        raise NotImplementedError()

    def __iter__(self):
        for i in range(self.dim1):
            for j in range(self.dim2):
                yield i, j, self.vertex[i, j]


class Susceptibilities(Integrable):
    i_iterator = 0

    def __init__(self):
        self.susceptibilities = []
        self.size = 0

    def append_all(self, name: str = None) -> object:
        if name is not None:
            _all_susc = name
        else:
            _all_susc = " ".join(
                list(
                    Susceptibility.SUSCEPTIBILITY_TYPE.keys()
                ))

        self.append_susceptibility_by_name(_all_susc)

        return self

    def append_susceptibility_by_name(self, name_string: str):
        susc_types = name_string.split()
        if len(susc_types) > 1:
            for name in susc_types:
                self.susceptibilities.append(
                    Susceptibility(name=name)
                )
                self.size += 1
        else:
            self.susceptibilities.append(
                Susceptibility(name=name_string)
            )
            self.size += 1

    def initialize(self, Np: int):
        self.Neq = 0
        for susc in self.susceptibilities:
            susc.initialize(Np=Np)
            self.Neq += susc.Neq

    def rg_equations(self, loops: Loops, interaction: Interaction):
        y = self.susceptibilities[0].rg_equations(loops, interaction)
        for i in range(1, self.size):
            y = np.concatenate(
                (y,
                 self.susceptibilities[i].rg_equations(loops, interaction))
            )
        return y

    def unpack(self, y: np.ndarray):
        indice = 0
        for i in range(self.size):
            neq = self.susceptibilities[i].Neq
            self.susceptibilities[i].unpack(
                y[indice:indice + neq]
            )
            indice += neq

    def initpack(self):
        y = self.susceptibilities[0].initpack()
        for i in range(1, self.size):
            y = np.concatenate(
                (y, self.susceptibilities[i].initpack())
            )
        return y

    def pack(self):
        pass

    def __iter__(self):
        for name in self.susceptibilities_types:
            yield self.susceptibilities[name]


if __name__ == "__main__":
    import time
    r = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32,
         "g1": 0.2, "g2": 0.2, "g3": 0.2}

    g = Interaction(r)
    b = Loops(r)
    b.get_values(lrg=1, Temperature=10)
    # print(b.Cooper)
    t1 = time.time()
    # for _ in range(10):
    #     g.rg_equations(b)
    # # input(time.time()-t1)

    # rf = Susceptibility("cdw")
    # rf.initialize(Np=r["Np"])

    #
    # y = rf.rg_equations(b, g)
    # print(rf.name)

    suscs = Susceptibilities()
    suscs.append_susceptibility_by_name("ssdw")
    suscs.initialize(Np=32)

    print(len(suscs.susceptibilities_types))
    for su in suscs:
        su.rg_equations(b, g)
        print(su.name)
        for i in su:
            print(i)
        print("================")
    # input(time.time()-t1)

    # for ty in rf.susceptibilities:
    #     rf2 = Susceptibility(ty)
    #     rf2.initialize(Np=r["Np"])
    #     print(ty)
    #     y = rf2.rg_equations(b, g)
    #     print(y)
