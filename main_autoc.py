"""Main script for auto-correlation tasks"""

from ising import simulator, plotter, thermo, datagen, loadingbar
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tasks import autoc


# autoc.generate(wipe=True, iternum=100, relaxtime=150)

# autoc.generate(wipe=False, iternum=100)

# autoc.generate(wipe=False, iternum=1000, bmin=0.4, bmax=0.6)

autoc.whatareks()

# autoc.analyse()
# autoc.results()
autoc.mosaics([13])

