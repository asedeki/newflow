import unittest
import sys
import pathlib
import numpy as np
import os
from random import choice, randint
path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow")+1])
else:
    path = "/".join(path.append("newflow"))

try:
    from loops import Loops
except ImportError:
    sys.path.append(path)
    from src.loops import Loops


class TestLoops(unittest.TestCase):
    def setUp(self):
        parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32}
        self.loops = Loops(**parameters)

    def test_init(self):
        self.loops.initialize(Temperature=10, lflow=1, tp=0, tp2=0)
        for k in self.loops.parameters:
            print(f"{k}  {self.loops.parameters[k]}")

    def test_call(self):

        Temperature = 1.0
        l_rg = 1.0
        self.loops.initialize(Temperature=Temperature, lflow=l_rg)
        self.loops()

        a = (np.sum(self.loops.Cooper), np.sum(self.loops.Peierls),
             np.sum(self.loops.Peierls_susc))
        b = (878.6685890125864, 878.5183879970207, 1.7964371318299874)
        #
        self.assertEqual(a, b)
        # # # # # input(loops.parameters)
        loops = self.loops
        Temperature = 10.0
        l_rg = 10.0
        loops.initialize(Temperature=Temperature, lflow=l_rg)
        loops()
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls), np.sum(loops.Peierls_susc))
        b = (2.5118151628507546, 2.6610631305629457, 0.007343458492368056)
        self.assertEqual(a, b)
        #
        loops = self.loops
        Temperature = 1e-10
        l_rg = 10.0
        self.loops.initialize(Temperature=Temperature)
        self.loops(l_rg)
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls), np.sum(loops.Peierls_susc))
        b = (37.949254425810715, 8.986533868805642, 0.015657680353671738)
        self.assertEqual(a, b)

        Temperature = 1e-80
        l_rg = 20.0
        loops.initialize(Temperature=Temperature, lflow=l_rg)
        loops()
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls), np.sum(loops.Peierls_susc))
        # # # input(loops.parameters)
        # print(a)
        b = (32.00127317596035, 0.0065097990057401195, 1.7901397210660127e-06)

        self.assertEqual(a, b)

        l_rg = 20.0
        loops.get_values(tp=0.0, tp2=0.0, Np=16, lflow=l_rg)
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls))
        Np = loops.parameters["Np"]
        b = (Np**2, Np**2)
        for i, j in zip(a, b):
            self.assertAlmostEqual(i, j)


if __name__ == "__main__":
    unittest.main()
