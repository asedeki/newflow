import unittest
import sys
import pathlib
import numpy as np
import os
import time
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
    from interaction import Interaction
except ImportError:
    sys.path.append(path)
    from src.loops import Loops
    from src.integrable import Integrable
    from src.dynamic import Dynamic
    from src.interaction import Interaction
    from src.interaction import Interaction


class TestInteraction(unittest.TestCase):
    # # def __init__(self):
    # #     self.integrable = Intg()

    @classmethod
    def setUpClass(cls):
        data = np.load("test_interactions.npy", allow_pickle=True)
        cls.T = data[()]["T"]
        dict_l_g = data[()]["g"]
        cls.g1 = {k: v[0] for k, v in dict_l_g.items()}
        cls.g2 = {k: v[1] for k, v in dict_l_g.items()}
        cls.g3 = {k: v[2] for k, v in dict_l_g.items()}

    # @classmethod
    # def tearDownClass(cls):
    #     pass

    # def setUp(self):
    #     pass

    # def tearDown(self):
    #     pass

    def test_integration_interaction(self):
        parameters = {
            "tp": 0, "tp2": 0,
            "Ef": 3000, "Np": 2,
            "g1": 0.2, "g2": 0.64, "g3": 0.0
        }
        loops = Loops()
        loops.initialize(**parameters, Temperature=1e-80)
        int = Interaction(parameters)
        int.set_loops(loops)
        integrator = Dynamic(
            rel_tol=1e-3).get_integrator(int)
        lf = 0.1
        li = 0

        g_dict = {}
        g_dict[0] = [int.g1[0, 0, 0], int.g2[0, 0, 0], int.g3[0, 0, 0]]
        for _ in range(1):
            integrator.next_value(l_ini=li, l_next=lf)
            g_dict[lf] = [int.g1[0, 0, 0], int.g2[0, 0, 0], int.g3[0, 0, 0]]
            #print(f"{loops.Temperature}\t{lf}\t {int.g1[0,0,0]}")
            self.assertAlmostEqual(int.g1[0, 0, 0], self.g1[lf], 4)
            self.assertAlmostEqual(int.g2[0, 0, 0], self.g2[lf], 4)
            self.assertAlmostEqual(int.g3[0, 0, 0], self.g3[lf], 4)
            li = lf
            lf += 0.1
        # data = {"T": loops.Temperature, "g": g_dict}
        # np.save("test_interactions", data)
        
    def test_integration_interaction_nm(self, data={}):

        tini = time.time()
        data = np.load("result.npy", allow_pickle=True)[()]

        parameters = data["param"]
        loops = Loops()
        loops.initialize(**parameters)
        int = Interaction(parameters)
        int.set_loops(loops)
        integrator = Dynamic(
            rel_tol=1e-5).get_integrator(int)
        
        li, lf = 0, 1
        integrator.next_value(l_ini=li, l_next=lf)
        data.update(
            {
                "param": parameters,
                "time":time.time()-tini,
                lf:[int.g1, int.g2, int.g3]
                }
            )
        self.assertAlmostEqual(np.sum(int.g1)-np.sum(data[lf][0]),0, NN)
        self.assertAlmostEqual(np.sum(int.g2)-np.sum(data[lf][1]),0, NN)
        self.assertAlmostEqual(np.sum(int.g3)-np.sum(data[lf][2]),0, NN)
        print(np.sum(int.g3)-np.sum(data[lf][2]))
        print(data["time"])
        np.save("data_test_interaction.npy", data)
NN=8

if __name__ == "__main__":
    unittest.main()
