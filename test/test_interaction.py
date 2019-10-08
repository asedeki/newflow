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

from src.interaction import Interaction
from src.dynamic import Dynamic
from src.integrable import Integrable
from src.loops import Loops

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

        intr = Interaction(**parameters)
        # intr.initialize(Temperature=1e-80)
        integrator = Dynamic(
            rel_tol=1e-4).get_integrator(intr)
        integrator.initialize(Temperature=1e-80)
        li, lf = 0.0, 0.1
        g_dict = {}
        g_dict[0] = [intr.g1[0, 0, 0], intr.g2[0, 0, 0], intr.g3[0, 0, 0]]
        g21 = 0.0
        Numb = 50
        for _ in range(Numb):
            integrator.next_value(l_ini=li, l_next=lf)
            g21 += 2*intr.g2[0, 0, 0]-intr.g1[0, 0, 0]
            g_dict[lf] = [intr.g1[0, 0, 0], intr.g2[0, 0, 0], intr.g3[0, 0, 0]]
            self.assertAlmostEqual(
                intr.g1[0, 0, 0], v_g(lf, parameters["g1"]), 4)
            self.assertAlmostEqual(
                intr.g2[0, 0, 0],
                parameters["g2"]-0.5*parameters["g1"] +
                0.5*v_g(lf, parameters["g1"]),
                4
            )

            lf, li = lf+0.1, lf
        self.assertAlmostEqual(g21/Numb, self.g12, 4)


NN = 10

if __name__ == "__main__":
    unittest.main()
