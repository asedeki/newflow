import os

import numpy as np

from newflow.plot import myplot

data_keys = ['susc', 'max_susc', 'interaction', 'time']
susc_keys = ['csdw', 'cbdw', 'ssdw', 'sbdw', 'supra_triplet', 'supra_singulet']
file = "data.npy"
data = np.load(file, allow_pickle=True)[()]

parameters = data["param"]
rel_tol = data["rel_tol"]
del data["rel_tol"]
del data["param"]
if "time" in data:
    del data["time"]
# input(parameters)
Temperatures = list(data.keys())
assert Temperatures[0] > Temperatures[-1]

times = []
pdfs = []
#max_susc = []
for T in Temperatures:
    #max_susc.append(data[T]["max_susc"][data[T]["max_susc"]["max"][0]])
    #times.append(data[T]["time"])
    interaction = data[T]#['interaction']
    gg = np.array(-interaction['g2']-interaction['g1'])
    pdfs.append(myplot.plot_pm3d(gg, T=T, z='$\Gamma_s$'))

#pdfs.insert(0, myplot.plot_1d(Temperatures, times, "time"))
#pdfs.insert(1, myplot.plot_1d(Temperatures, max_susc,
#                              f"{data[T]['max_susc']['max'][0]}"))
pdf = " ".join(pdfs)
outpdf = file.split(".npy")[0]
os.system(f"pdfunite {pdf} {outpdf}.pdf")
os.system(f"rm -f {pdf}")
