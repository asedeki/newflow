import os
import sys
import time

import numpy as np
path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path.append("newflow")
    path = "/".join(path)
sys.path.append(path)


import src.dynamic as dyn
import src.interaction as sint
import src.loops as lp

# from random import choice, randint


def main(data={}):
    parameters = {
        "tp": 200, "tp2": 32,
        "Ef": 3000, "Np": 64,
        "g1": 0.32, "g2": 0.64, "g3": 0.03
    }
    Temperatures = np.linspace(100, 10, 10)
    tini = time.time()
    loops = lp.Loops()
    loops.initialize(**parameters, Temperature=-1)
    int = sint.Interaction(parameters)
    int.set_loops(loops)

    integrator = dyn.Dynamic(
        rel_tol=1e-3).get_integrator(int)

    for i in range(len(Temperatures)):
        integrator.initialize(Temperature=Temperatures[i])
        li, lf = 0, 50
        if integrator.next_value(l_ini=li, l_next=lf):
            print(f"""
              T= {loops.parameters["Temperature"]}
              max_g = {max(np.max(int.g1),
                np.max(int.g2), np.max(int.g3))}
              """)
            data.update(
                {
                    Temperatures[i]: [int.g1, int.g2, int.g3]
                }
            )
    data["param"] = parameters
    data["time"] = time.time() - tini
    print(data["time"])

    #np.save("Tevol_interaction.npy", data)


main()
