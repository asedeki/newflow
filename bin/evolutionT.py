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

# from random import choice, randint


def main(data={}):
    parameters = {
        "tp": 200, "tp2": 27,
        "Ef": 3000, "Np": 32,
        "g1": 0.32, "g2": 0.64, "g3": 0.02
    }
    Temperatures = np.concatenate(
        (
            np.linspace(100, 10, 10),
            np.linspace(9, 1, 9)
        )
    )
    print(Temperatures)

    int = sint.Interaction(**parameters)
    data = {
        "param": parameters
    }
    integrator = dyn.Dynamic(
        rel_tol=1e-3).get_integrator(int, ode_method='dopri5')

    for T in Temperatures:
        print(f"T={T}")
        tini = time.time()
        integrator.initialize(Temperature=T)
        li, lf = 0, 50
        if integrator.next_value(l_ini=li, l_next=lf):

            data.update(
                {
                    T: {
                        'g1': int.g1,
                        'g2': int.g2,
                        'g3': int.g3
                    },
                    'time': time.time() - tini
                }
            )
        np.save(f"data_interaction_tp2_{parameters['tp2']}.npy", data)


main()
