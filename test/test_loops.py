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
    # def setUpClass(cls):
    #     pass

    # def tearDownClass(self):
    #     pass

    def setUp(self):
        self.parameters = {
            "tp": 200, "tp2": 20,
            "Ef": 3000, "Np": 32
        }
        self.loops = Loops(self.parameters)

    def tearDown(self):
        pass

    def test_init(self):
        keys = ["tp", "tp2", "Ef"]
        n = randint(1, 2)
        not_miss = set([])
        for _ in range(n):
            not_miss.add(choice(keys))
        miss = []
        for key in keys:
            if key not in not_miss:
                del self.parameters[key]
                miss.append(key)

        miss.sort()

        message = "{} must have {} keys".format(
            self.parameters.__str__(), ','.join(miss))
        try:
            self.assertRaisesRegex(KeyError, message,
                                   Loops, self.parameters)
        except AssertionError:
            print(f"message attendu = {message}")
            print(f"parametres loops = {Loops.parameters}")
            raise AssertionError
        parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32}
        result = {'tp': 200, 'tp2': 20, 'Ef': 3000,
                  'Np': 32, 'Temperature': 1e-80, "lrg": None}

        loops = Loops(parameters)
        self.assertDictEqual(result, loops.parameters)

    def test_call(self):
        parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32}
        Temperature = 1.0
        l_rg = 1.0
        self.loops.initialize(**parameters,
                              Temperature=Temperature,
                              lrg=l_rg)
        self.loops()
        a = (np.sum(self.loops.Cooper), np.sum(self.loops.Peierls),
             np.sum(self.loops.Peierls_susc))
        b = (878.6685890125864, 878.5183879970207, 1.7964371318299874)

        self.assertEqual(a, b)
        # # # # input(loops.parameters)
        loops = self.loops
        Temperature = 10.0
        l_rg = 10.0
        loops.initialize(Temperature=Temperature, lrg=l_rg)
        loops()
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls), np.sum(loops.Peierls_susc))
        b = (2.5118151628507546, 2.6610631305629457, 0.007343458492368056)
        self.assertEqual(a, b)

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
        loops.initialize(Temperature=Temperature, lrg=l_rg)
        loops()
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls), np.sum(loops.Peierls_susc))
        # # # input(loops.parameters)
        # print(a)
        b = (32.00127317596035, 0.0065097990057401195, 1.7901397210660127e-06)
        self.assertEqual(loops.Temperature, 1e-80)
        self.assertEqual(a, b)

        l_rg = 20.0
        loops.get_values(tp=0.0, tp2=0.0, Np=16, lrg=l_rg)
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls))
        Np = loops.parameters["Np"]
        b = (Np**2, Np**2)
        self.assertEqual(loops.Temperature, 1e-80)
        for i, j in zip(a, b):
            self.assertAlmostEqual(i, j)


if __name__ == "__main__":
    unittest.main()
