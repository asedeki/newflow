import unittest
import sys
import pathlib
import numpy as np
# from random import choice, randint

try:
    from ..src.loops import Loops
    from ..src.integrable import Integrable
    from ..src.dynamic import Dynamic
    from ..src.integrablesystem import IntegrableSystem
except ImportError:
    python_path = pathlib.posixpath.abspath("..")
    # # input(python_path)
    sys.path.append(python_path)
    from src.loops import Loops
    from src.integrable import Integrable
    from src.dynamic import Dynamic
    from src.integrablesystem import IntegrableSystem
except ValueError:
    python_path = pathlib.posixpath.abspath(".")
    sys.path.append(python_path)
    from newflow.src.loops import Loops
    from newflow.src.integrable import Integrable
    from newflow.src.dynamic import Dynamic
    from newflow.src.integrablesystem import IntegrableSystem


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

    def rg_equations(self, y: np.ndarray, lflow: float):
        self.unpack(y)
        dy0 = self.y[0]
        dy1 = 1.0/self.y[1]
        # # input(f"{dy0}, {dy1}")
        return self.pack(dy0, dy1)


class TestLoops(unittest.TestCase):
    # def __init__(self):
    #     self.integrable = Intg()

    @classmethod
    def setUpClass(cls):
        parameters = {
            "tp": 200, "tp2": 20,
            "Ef": 3000, "Np": 4
        }
        #loops = Loops(parameters)
        I1 = Intg(1, 1)
        I2 = Intg(2, 2)
        cls.I1 = I1
        # cls.dyn_sys = IntegrableSystem(parameters=parameters)
        # cls.dyn_sys.set_interaction(I1)
        # cls.dyn_sys.set_suscptibilities({"1": I1, "2": I2})
        # cls.dyn_sys.set_loops(loops=loops)
        # cls.d = Dynamic(
        #     rel_tol=1e-3).get_integrator(cls.dyn_sys)
        cls.d = Dynamic(
            rel_tol=1e-3).get_integrator(I1)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_integrator(self):
        lf = 1
        li = 0
        for _ in range(1):
            y0 = self.I1.y[0]
            self.d.next_value(l_ini=li, l_next=lf)
            #self.d.evolutionl(0, l)
            self.assertAlmostEqual(self.I1.y[0], y0*np.exp(lf-li))
            li = lf
            lf += 0.6
    # def test_init(self):
    #     self.d.initialize()
    #     for intg_ in self.d.systems:
    #         self.assertTrue(
    #             np.all(intg_.loops.Cooper == Intg.loops.Cooper)
    #         )
    #         self.assertTrue(
    #             np.all(intg_.loops.Peierls == Intg.loops.Peierls)
    #         )
    #         self.assertTrue(
    #             np.all(intg_.loops.Peierls_susc == Intg.loops.Peierls_susc)
    #         )

    # def test_g(self):
    #     parameters = {
    #         "tp": 200, "tp2": 20,
    #         "Ef": 3000, "Np": 4
    #     }
    #     loops = Loops(parameters)
    #     loops.initialize(tp2=0.0, tp=0.0)
    #     Integrable.loops = loops
    #     I1 = Intg(1, 1)
    #     I2 = Intg(2, 2)
    #     print(I1.goology)
    #     print(I2.goology)


if __name__ == "__main__":
    unittest.main()
