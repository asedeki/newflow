import unittest
import sys
import pathlib
import numpy as np
import os
# from random import choice, randint
path = os.getcwd().split("/")
if "newflow" in path:
    path = "/".join(path[:path.index("newflow")+1])
else:
    path = "/".join(path.append("newflow"))

try:
    from loops import Loops
    from integrable import Integrable
    from dynamic import Dynamic
    from integrablesystem import IntegrableSystem
except ImportError:
    python_path = pathlib.posixpath.abspath("..")
    # # input(python_path)
    sys.path.append(python_path)
    from src.loops import Loops
    from src.integrable import Integrable
    from src.dynamic import Dynamic
    from src.integrablesystem import IntegrableSystem


class Intg(Integrable):
    def __init__(self, *vargs):
        self.y = np.zeros(2, float)
        self.v = vargs
        self.Neq = 2
        super().__init__()

    def initialize(self, *vargs):
        self.y = np.array(list(self.v))

    def initpack(self, *vargs):
        return self.y

    def pack(self, *vargs):
        return np.array(vargs)

    def unpack(self, y):
        self.y = y

    def rg_equations(self, lflow: float = None):

        dy1 = self.y[1]
        dy0 = 1.0/self.y[0]/2.0
        return self.pack(dy0, dy1)


class TestLoops(unittest.TestCase):
    def test_integrator(self):
        I1 = Intg(1, 1)
        I1.initialize()
        d = Dynamic(
            rel_tol=1e-10).get_integrator(I1)
        lf = 10
        li = 0
        for _ in range(20):
            y0 = I1.y[0]
            y1 = I1.y[1]
            d.next_value(l_ini=li, l_next=lf)

            #self.d.evolutionl(0, l)
            # print(f"l_final, l_ini = {lf},{li}")
            # print("_____________________________")
            self.assertAlmostEqual(I1.y[1], y1*np.exp(lf-li), 4)
            self.assertAlmostEqual(I1.y[0], np.sqrt(lf-li+y0**2), 4)
            li = lf
            lf += 0.4

    def test_integrable_system(self):
        I1 = Intg(1, 1)
        I2 = Intg(2, 2)
        S = IntegrableSystem()
        S.add_integrable_system(I1, I2)
        S.initialize()
        d = Dynamic(
            rel_tol=1e-10).get_integrator(S)
        lf = 10
        li = 0
        for _ in range(20):
            y0 = I2.y[0]
            y1 = I2.y[1]
            d.next_value(l_ini=li, l_next=lf)
            #self.d.evolutionl(0, l)
            # print(f"l_final, l_ini = {lf},{li}")
            # print("_____________________________")
            self.assertAlmostEqual(I2.y[1], y1*np.exp(lf-li), 4)
            self.assertAlmostEqual(I2.y[0], np.sqrt(lf-li+y0**2), 4)
            li = lf
            lf += 0.4


if __name__ == "__main__":
    unittest.main()
