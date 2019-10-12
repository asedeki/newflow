import os
import pathlib
import numpy as np
import newflow.plot.myplot as myplot

if __name__ == "__main__":
    path = pathlib.Path()
    files = path.rglob('*.npy')
    result = {}
    for f in files:
        print(f)
        result = np.load(f, allow_pickle=True)[()]
        Temp = list(result.keys())
        pdfs = []
        for T in Temp:
            if isinstance(T, float):
                print(T)
                print(result[T]["susc"]["SSDW_pi"])
                print(result[T]["susc"]["SS_dx2y2"])
                print('==============')


                g = np.array(-result[T]['interaction']['g2'])
                # if T == 5.0:
                pdfs.append(myplot.plot_pm3d(g, Temperature=T))
        pdf = " ".join(pdfs)
        os.system(f"pdfunite {pdf} out_{f}.pdf")
        os.system(f"rm -f {pdf}")
