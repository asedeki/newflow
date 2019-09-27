import unittest
import sys
import pathlib
import numpy as np
import os
# from random import choice, randint
path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow") + 1])
else:
    path = "/".join(path.append("newflow"))

try:
    from loops import Loops
    from integrable import Integrable
    from dynamic import Dynamic
    from integrablesystem import IntegrableQuasi1dSystem
except ImportError:
    sys.path.append(path)
    from src.loops import Loops
    from src.integrable import Integrable
    from src.dynamic import Dynamic
    from src.integrablesystem import IntegrableQuasi1dSystem
    from src.interaction import Interaction
    from src.quasi1dsusceptibilities import Susceptibilities


class TestIntegrableQuasi1dSystem(unittest.TestCase):
    # def __init__(self):
    #     self.integrable = Intg()

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_integration_interaction(self):
        parameters = {
            "tp": 200, "tp2": 20,
            "Ef": 3000, "Np": 32,
            "g1": 0.2, "g2": 0.64, "g3": 0.0,
            "Temperature": 100
        }

        IQ = IntegrableQuasi1dSystem().set_all(
            parameters=parameters
        )
        # input(IQ.Neq)
        integrator = Dynamic(
            rel_tol=1e-3).get_integrator(IQ)
        lf = 0.1
        li = 0
        integrator.next_value(l_ini=li, l_next=lf)
        #
        # g_dict = {}
        # g_dict[0] = [int.g1[0, 0, 0], int.g2[0, 0, 0], int.g3[0, 0, 0]]
        # for _ in range(1):
        #     integrator.next_value(l_ini=li, l_next=lf)
        #     g_dict[lf] = [int.g1[0, 0, 0], int.g2[0, 0, 0], int.g3[0, 0, 0]]
        #     print(f"{loops.Temperature}\t{lf}\t {int.g1[0,0,0]}")
        #     li = lf
        #     lf += 0.1
        # data = {"T": loops.Temperature, "g": g_dict}
        # np.save("test_interactions", data)

    # def test_integration_quasi1d(self):
    #     parameters = {
    #         "tp": 0, "tp2": 0,
    #         "Ef": 3000, "Np": 4,
    #         "g1": 0.2, "g2": 0.64, "g3": 0.0
    #     }

    #     q1ds = Quasi1dSystemparameters=parameters).set_all()

    #     integrator = Dynamic(
    #         rel_tol=1e-3).get_integrator(q1ds)
    #     lf = 0.1
    #     li = 0
    #     integrator.next_value(l_ini=li, l_next=lf)
    #     g_dict = {}
    #     g_dict[0] = [int.g1[0, 0, 0], int.g2[0, 0, 0], int.g3[0, 0, 0]]
    #     for _ in range(100):
    #         integrator.next_value(l_ini=li, l_next=lf)
    #         g_dict[lf] = [int.g1[0, 0, 0], int.g2[0, 0, 0], int.g3[0, 0, 0]]
    #         print(f"{loops.Temperature}\t{lf}\t {int.g1[0,0,0]}")
    #         li = lf
    #         lf += 0.1
    #     data = {"T": loops.Temperature, "g": g_dict}
    #     np.save("test_interactions", data)


if __name__ == "__main__":
    unittest.main()
