import unittest
import sys
import numpy as np
import os
import time
# from random import choice, randint
path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path.append("newflow")
    path = "/".join(path)
sys.path.append(path)

from src.loops import Loops
from src.integrable import Integrable
from src.dynamic import Dynamic
from src.interaction import Interaction
from src.interaction import Interaction


def v_g(l, g1):
    return g1/(g1*l+1)

class TestInteraction(unittest.TestCase):
    # # def __init__(self):
    # #     self.integrable = Intg()

    @classmethod
    def setUpClass(cls):
        pass
    # @classmethod
    # def tearDownClass(cls):
    #     pass

    # def setUp(self):
    #     pass

    # def tearDown(self):
    #     pass

    def test_integration_interaction_T0(self):
        parameters = {
            "tp": 0, "tp2": 0,
            "Ef": 3000, "Np": 2,
            "g1": 0.2, "g2": 0.0, "g3": 0.0
        }
        self.g12 = 2*parameters["g2"]-parameters["g1"]

        loops = Loops()
        loops.initialize(**parameters, Temperature=1e-80)
        int = Interaction(parameters)
        int.set_loops(loops)

        integrator = Dynamic(
            rel_tol=1e-6).get_integrator(int)
        integrator.initialize(**parameters, Temperature=1e-80)
        li, lf = 0.0, 0.1
        g_dict = {}
        g_dict[0] = [int.g1[0, 0, 0], int.g2[0, 0, 0], int.g3[0, 0, 0]]
        g21 = 0.0
        Numb = 50
        for _ in range(Numb):
            integrator.next_value(l_ini=li, l_next=lf)
            g21 += 2*int.g2[0, 0, 0]-int.g1[0, 0, 0]
            g_dict[lf] = [int.g1[0, 0, 0], int.g2[0, 0, 0], int.g3[0, 0, 0]]
            self.assertAlmostEqual(
                int.g1[0, 0, 0], v_g(lf, parameters["g1"]), 4)
            self.assertAlmostEqual(
                int.g2[0, 0, 0],
                parameters["g2"]-0.5*parameters["g1"] +
                0.5*v_g(lf, parameters["g1"]),
                4
            )

            lf, li = lf+0.1, lf
        self.assertAlmostEqual(g21/Numb, self.g12, 4)

    def test_integration_interaction_l(self, data={}):

        tini = time.time()
        data = np.load("data_test_interaction.npy", allow_pickle=True)[()]
        parameters = data["param"]
        loops = Loops()
        loops.initialize(**parameters)
        int = Interaction(parameters)
        int.set_loops(loops)
        li, lf = 0, 1
        # integrator = Dynamic(rel_tol=1e-5)
        # integrator.set_dynamical_system(int)
        #integrator.evolutionl(li, lf)

        integrator = Dynamic(
            rel_tol=1e-5).get_integrator(int)
        integrator.initialize()
        integrator.next_value(l_ini=li, l_next=lf)

        data.update(
            {
                "param": parameters,
                "time": time.time()-tini,
                lf: [int.g1, int.g2, int.g3]
            }
        )

        self.assertAlmostEqual(np.sum(int.g1)-np.sum(data[lf][0]), 0, NN)
        self.assertAlmostEqual(np.sum(int.g2)-np.sum(data[lf][1]), 0, NN)
        self.assertAlmostEqual(np.sum(int.g3)-np.sum(data[lf][2]), 0, NN)
        print(np.sum(int.g3)-np.sum(data[lf][2]))
        print(data["time"])

        np.savez("data_test_interaction.npy", data)

    # def test_integration_interaction_T(self, data={}):

    #     # TODO: probleme d'initialization
    #     # wrning: probleme initialization tout court!!!!

    #     parameters = {
    #         "tp": 200, "tp2": 30,
    #         "Ef": 3000, "Np": 16,
    #         "g1": 0.32, "g2": 0.64, "g3": 0.03
    #     }
    #     Temperatures = np.linspace(100, 10, 10)
    #     tini = time.time()
    #     loops = Loops()
    #     loops.initialize(**parameters, Temperature=-1)
    #     int = Interaction(parameters)
    #     int.set_loops(loops)
    #     d = Dynamic(
    #         rel_tol=1e-3)

    #     for i in range(len(Temperatures)):
    #         integrator = d.get_integrator(int)
    #         print(np.sum(int.g1))
    #         li, lf = 0, 1
    #         integrator.next_value(l_ini=li, l_next=lf,
    #                               Temperature=Temperatures[i])
    #         print(np.sum(int.g1))
    #         input(d.dynamical_sytem.loops.parameters["Temperature"])
    #         data.update(
    #             {
    #                 Temperatures[i]: [int.g1, int.g2, int.g3]
    #             }
    #         )
    #     data["param"] = parameters
    #     data["time"] = time.time()-tini
    #     print(data["time"])

    #     np.save("data_test_interaction.npy", data)


NN = 8

if __name__ == "__main__":
    unittest.main()
