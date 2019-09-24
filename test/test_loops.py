import unittest
import sys
import pathlib
import numpy as np
from random import choice, randint

try:
    from ..src.loops import Loops
except ImportError:
    python_path = pathlib.posixpath.abspath("..")
    # input(python_path)
    sys.path.append(python_path)
    from src.loops import Loops
except ValueError:
    python_path = pathlib.posixpath.abspath(".")
    sys.path.append(python_path)
    from newflow.src.loops import Loops


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

    def tearDown(self):
        pass

    def test_init(self):
        keys = list(self.parameters.keys())
        n = randint(1, len(keys)+1)
        not_miss = set([])
        for _ in range(n):
            not_miss.add(choice(keys))
        miss = []
        for key in keys:
            if key not in not_miss:
                del self.parameters[key]
                miss.append(key)

        miss.sort()
        # input(miss)
        # input(self.parameters)
        message = "{} must have {} keys".format(
            self.parameters.__str__(), ','.join(miss))
        self.assertRaisesRegex(KeyError, message,
                               Loops, self.parameters)

        parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32}
        result = {'tp': 200, 'tp2': 20, 'Ef': 3000,
                  'Np': 32, 'Temperature': 1e-80}
        loops = Loops(parameters)
        self.assertDictEqual(result, loops.param)

    def test_call(self):
        parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32}
        loops = Loops(parameters)
        Temperature = 1.0
        l_rg = 1.0
        # loops.initialize(Temperature=Temperature)
        # loops(l_rg)
        loops.initialize(Temperature=Temperature)(l_rg)
        a = (np.sum(loops.Cooper), np.sum(loops.Peierls),
             np.sum(loops.Peierls_susc))
        b = (878.6685890125864, 878.5183879970207, 1.7964371318299874)

        self.assertEqual(a, b)
        # input(loops.param)

        Temperature = 10.0
        l_rg = 10.0
        loops.loops_integration_donne = False
        loops.initialize(Temperature=Temperature)
        loops(l_rg)
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls), np.sum(loops.Peierls_susc))
        b = (2.5118151628507546, 2.6610631305629457, 0.007343458492368056)
        self.assertEqual(a, b)
        # input(loops.param)

        Temperature = 1e-10
        l_rg = 10.0
        loops.initialize(Temperature=Temperature)
        loops.loops_integration_donne = False
        loops(l_rg)
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls), np.sum(loops.Peierls_susc))
        b = (37.949254425810715, 8.986533868805642, 0.015657680353671738)
        self.assertEqual(a, b)

        l_rg = 20.0
        loops.initialize()
        loops.loops_integration_donne = False
        loops(l_rg)
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls), np.sum(loops.Peierls_susc))
        # input(loops.param)
        # print(a)
        b = (32.00127317596035, 0.0065097990057401195, 1.7901397210660127e-06)
        self.assertEqual(loops.Temperature, 1e-80)
        self.assertEqual(a, b)

        l_rg = 20.0
        loops.initialize(tp=0.0, tp2=0.0, Np=8)
        loops.loops_integration_donne = False
        loops(l_rg)
        a = (np.sum(loops.Cooper), np.sum(
            loops.Peierls))
        Np = loops.param["Np"]
        b = (Np**2, Np**2)
        self.assertEqual(loops.Temperature, 1e-80)
        for i, j in zip(a, b):
            self.assertAlmostEqual(i, j)


if __name__ == "__main__":
    unittest.main()
