import os
import sys
import time

import numpy as np

path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path = "/".join(path.append("newflow"))
sys.path.append(path)

from src.dynamic import Dynamic
from src.integrablesystem import Quasi1dIntegrableSystem as QIS


parameters = {
    "tp": 200, "tp2":32,
    "Ef": 3000, "Np": 32,
    "g1": 0.32, "g2": 0.64, "g3": 0.02
}

qis = QIS().set_all(**parameters)

# IS = IntegrableSystem()
# IS.add_integrable_system(Int)

rel_tol = 1e-4

d = Dynamic(rel_tol=rel_tol)  # .get_integrator()
d.set_dynamical_system(qis)


integrer = True
T = 5.0
dT = 1.0
data = {}
data["param"] = parameters
data["rel_tol"] = rel_tol
t0 = time.time()
while integrer:
    t1 = time.time()
    print(f'T = {T}')
    d.initialize(Temperature=T)
    if d.next(l_ini=0, l_next=100):
        # d.next(l_ini=0, l_next=100)
        data[T] = {
            "susc": qis.susceptibilities.values,
            "max_susc": qis.susceptibilities.maximum,
            "interaction": {
                "g1": qis.interaction.g1,
                "g2": qis.interaction.g2,
                "g3": qis.interaction.g3
            },
            "time": time.time() - t1
        }
        np.save(f"data_tp2_{parameters['tp2']}", data)
        T = T - dT
        if T <= 0.0 :
            dT = dT/10.0
            T = 9.0*dT
        if T <= 0.1:
            integrer = False
    else:
        if dT == 0.1:
            integrer = False
        else:
            dT = dT/10.0
            T += 9.0 * dT

t2 = time.time()
print(f"t_now = {t2 - t0}")
