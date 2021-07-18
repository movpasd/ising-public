"""Main script for relaxation task"""

from ising import simulator, plotter, thermo, datagen, loadingbar
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tasks import relaxation


datapath = Path(__file__).parents[0] / "data/relaxation"
resultspath = Path(__file__).parents[0] / "results/relaxation"

# Generation and analysis of data
# ------------------------------------------------------------------------

# relaxation.generate(datapath)

# mags, sqmags, diffs, smoothed_diffs = relaxation.analyse(datapath)
# np.save(datapath / "mags.npy", mags)
# np.save(datapath / "sqmags.npy", sqmags)
# np.save(datapath / "diffs.npy", diffs)
# np.save(datapath / "smoothed_diffs.npy", smoothed_diffs)

# Loading the data (only once it's generated!)
# ------------------------------------------------------------------------

# aligned_dataset = datagen.DataSet(datapath / "init_aligned", load=True)
# random_dataset = datagen.DataSet(datapath / "init_random", load=True)

# Loading pre-calculated data
# ------------------------------------------------------------------------

a_mags, r_mags = np.load(datapath / "mags.npy")
a_sqmags, r_sqmags = np.load(datapath / "sqmags.npy")
a_diffs, r_diffs = np.load(datapath / "diffs.npy")
a_smoothed_diffs, r_smoothed_diffs = np.load(
    datapath / "smoothed_diffs.npy")
a_autocs, r_autocs = np.load(datapath / "autocs.npy")

# Drawing pretty magnetisation graphs
# ------------------------------------------------------------------------

sqmags = [a_sqmags, r_sqmags]
diffs = [a_diffs, r_diffs]
smoothed_diffs = [a_smoothed_diffs, r_smoothed_diffs]

relaxation.graph(resultspath, sqmags, diffs, smoothed_diffs,
                 save=True, show=False)

# Generating mosaic animations
# ------------------------------------------------------------------------

# for k in range(11):
#     print(f"Animating r-b{k}")
#     _, _, anim = plotter.animate_mosaic(random_dataset.ensembles[k],
#                                     timestamp=True, verbose=True)
#     anim.save(str(resultspath / f"mosaic-r-b{k}.mp4"))

#     print(f"Animating a-b{k}")
#     _, _, anim = plotter.animate_mosaic(aligned_dataset.ensembles[k],
#                                     timestamp=True, verbose=True)
#     anim.save(str(resultspath / f"mosaic-a-b{k}.mp4"))

# Calculate the time to equilibrium
# ------------------------------------------------------------------------

# The criterion for equilibrium is that the average change in the (square)
# magnetisation drops below some cutoff. Heuristically after looking at
# both the mosaic plots and the smoothed_diffs graph I decided to use
# a cutoff of 0.001.

# This is far from perfect for the initially random spins with low temps
# due to the formation of stable domains. So really for those I've
# measured the time to metastable domained equilibrium rather than
# true equilibrium where the whole grid goes to up or down.

# cutoff = 0.001

# bs = np.array(range(0, 11)) / 10

# r_cutoffs = np.argmax(np.abs(r_smoothed_diffs) < cutoff, axis=-1)
# a_cutoffs = np.argmax(np.abs(a_smoothed_diffs) < cutoff, axis=-1)

# print("random", r_cutoffs)
# print("aligned", a_cutoffs)

# plt.close()

# plt.plot(bs, r_cutoffs, "r+", markersize=7)
# plt.plot(bs, a_cutoffs, "b+", markersize=7)

# plt.legend(["init. random", "init. aligned"])

# plt.xlabel("$ \\beta $")
# plt.ylabel("$ \\tau $")
# plt.title("Temperature dependence of relaxation time, 30x30 grid")

# plt.savefig(resultspath / "relaxtime.pdf")
# plt.show()
