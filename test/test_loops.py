import unittest
import sys
import pathlib
import numpy as np
import os
from random import choice, randint
import asyncio
import concurrent
# try:
from newflow.src.loops import Loops
# except Exception:
#     print('in exception import')
#     path = os.getcwd().split("/")
#     if "newflow" in path:
#         path = "/".join(path[:path.index("newflow") + 1])
#     else:
#         path = "/".join(path.append("newflow"))

#     sys.path.append(path)
#     print(sys.path)
#     from src.loops import Loops


def calc(l):
    # print(l)
    return l


class TestLoops(unittest.TestCase):
    def setUp(self):
        parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 16}
        self.loops = Loops(**parameters)

    # def test_init(self):
    #     self.loops.initialize(Temperature=10, lflow=1, tp=0, tp2=0)
    #     for k in self.loops.parameters:
    #         print(f"{k}  {self.loops.parameters[k]}")
    # @staticmethod
    async def calcul(self, lrg):
        print(lrg)
        parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32}
        loops = Loops(**parameters)
        loops.initialize(Temperature=1, lflow=lrg)
        loops()
        print(np.sum(loops.Cooper))
        return np.sum(loops.Cooper)
    def calcul2(self, lrg):
        print(lrg)
        parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32}
        loops = Loops(**parameters)
        loops.initialize(Temperature=1, lflow=lrg)
        loops()
        print(np.sum(loops.Cooper))
        return np.sum(loops.Cooper)
            
    def test_temps(self):
        import time
        import numpy
        # ll = numpy.linspace(1,50,20)
        # t1 = time.time()
        # async def main():
        #     tasks = []
        #     for l in ll:
        #         task = asyncio.ensure_future(self.calcul(l))
        #         tasks.append(task)
        #     await asyncio.gather(*tasks)
        # loop = asyncio.get_event_loop()
        
        # try:
        #     loop.run_until_complete(main())
        # finally:
        #     loop.close()
        # for tt in tasks:
        #     print(tt.get_values())
        t1=time.time()
        # print(f't_loop = {t2 - t1}')
        
        def bcalcul(l):
            parameters = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 32}
            loops = Loops(**parameters)
            loops.initialize(Temperature=1, lflow=l)
            loops()
            return l, np.sum(loops.Cooper)
        ll = numpy.linspace(1,50,10)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            exx = executor.map(bcalcul, ll)
        val ={}
        for l,v in exx:
            val[l] = v
        t2=time.time()
        t_loop = t2 - t1
        for l in ll:
            self.assertEqual(val[l]-bcalcul(l)[1],0)

        # self.assertGreater(time.time() - t2, 2*t_loop)
        
    def atest_call(self):
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
