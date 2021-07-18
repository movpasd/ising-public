"""Calculation of mean magnetisation variation with temperature"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from warnings import warn

from ising import datagen, loadingbar, plotter, simulator, thermo


# In this task I re-use the data from tasks.autoc, except I only use
# N = 30 and include values for b far from critical.


autocdatapath = Path(__file__).parents[1] / "data/autoc-night"
datapath = Path(__file__).parents[1] / "data/meanmag"
resultspath = Path(__file__).parents[1] / "results/meanmag"


def create_dataset():
    """Creates the mainmag dataset from the autoc dataset"""

    print("Loading autoc data")
    autocdataset = datagen.DataSet(autocdatapath)
    autocdataset.load()

    print("Preparing new dataset for copying")
    newdataset = datagen.DataSet(datapath)
    newdataset.wipe()

    newk = 0

    for k, ens in enumerate(autocdataset.ensembles):

        N, b = ens.grid_shape[0], ens.b

        if N == 30:

            print(f"Adding {k} as {newk} | "
                  f"N={N} b={b:.2f} t={ens.iternum}")

            newk += 1
            newdataset.add_ensemble(ens, save=True)


def randflip():
    """Randomly flip the systems"""

    # See tasks.autoc.randflip() for explanation

    dataset = datagen.DataSet(datapath)
    dataset.load()

    for k, ens in enumerate(dataset.ensembles):

        print(k)
        ens.do_randflip()
        dataset.save(k)


def _load_dataset():
    """Loads dataset and figures out the k<-->b correspondence"""

    dataset = datagen.DataSet(datapath)
    dataset.load()

    ensembles = dataset.ensembles
    bs = [ens.b for ens in ensembles]

    # Verify they all have size N=30
    for ens in ensembles:
        if ens.grid_shape[0] != ens.grid_shape[1]:
            raise ValueError("Bad grid shape found")
        if ens.grid_shape[0] != 30:
            raise ValueError("Bad grid shape found")

    return dataset, ensembles, bs


def analyse():

    print("Analysing dataset")

    dataset, ensembles, bs = _load_dataset()

    np.save(datapath / "bs.npy", np.array(bs))

    # Indexing: 0-axis will label ensemble, 1-axis will label system
    # These are all time-averaged quantities
    mags = []
    sqmags = []
    flucts_mag = []
    flucts_sqmag = []

    for k, ens in enumerate(ensembles):

        print(f"{k} | b = {ens.b:.2f}")

        arr = ens.asarray()
        magarr = thermo.magnetisation(arr)
        sqmagarr = thermo.square_mag(arr)
        time = 0  # the time axis

        mags.append(np.mean(magarr, axis=time))
        sqmags.append(np.mean(sqmagarr, axis=time))
        flucts_mag.append(np.std(magarr, axis=time, ddof=1))
        flucts_sqmag.append(np.std(sqmagarr, axis=time, ddof=1))

    np.save(datapath / "mags.npy", np.stack(mags, axis=0))
    np.save(datapath / "sqmags.npy", np.stack(sqmags, axis=0))
    np.save(datapath / "flucts_mag.npy", np.stack(flucts_mag, axis=0))
    np.save(datapath / "flucts_sqmag.npy", np.stack(flucts_sqmag, axis=0))


def results():

    print("Drawing up results")

    # Load up all the data

    bs = np.load(datapath / "bs.npy")

    mags = np.load(datapath / "mags.npy")
    sqmags = np.load(datapath / "sqmags.npy")
    flucts_mag = np.load(datapath / "flucts_mag.npy")
    flucts_sqmag = np.load(datapath / "flucts_sqmag.npy")

    sysnum = mags.shape[1]

    # 1. Branching magnetisation plot
    # -------------------------------

    plt.figure(figsize=(8, 6))

    # Unpack all the ensembles and just scatter plot the time-averaged
    # magnetisations of each system against temperature
    plt.plot(np.repeat(bs[:, np.newaxis], sysnum, axis=1).flatten(),
             mags.flatten(), "r.", ms=1)

    plt.title("Time-averaged magnetisation of a collection of systems")
    plt.xlabel("$\\beta$")
    plt.ylabel("$ \\langle M \\rangle_t $")

    plt.savefig(resultspath / "branches.pdf")
    # plt.show()
    plt.close()

    # 2. Square magnetisation plot
    # ----------------------------

    # For this plot, we take the ensemble average of the square magnetisation
    # values, which were already time averaged
    #   I do the square magnetisation because the raw magnetisation would
    # average to zero due to symmetry.
    #   Estimated error using standard deviation of ensemble

    # kwargs for errorbar
    ebar_kw = {
        "fmt": "kx",
        "ecolor": "r",
        "ms": 10,
        "elinewidth": 3,
    }

    plt.figure(figsize=(8, 6))

    # ensemble average
    values = np.mean(sqmags, axis=1)
    errors = np.std(sqmags, axis=1, ddof=1) / np.sqrt(sysnum)

    plt.errorbar(bs, values, errors, **ebar_kw)

    plt.title("Square magnetisation versus temperature")
    plt.xlabel("$\\beta$")
    plt.ylabel("$ \\langle M^2 \\rangle_{t, e} $")

    plt.savefig(resultspath / "sqmag.pdf")
    # plt.show()
    plt.close()

    # 3. Fluctuations plot
    # --------------------

    # This displays the time-fluctuations of the magnetisation versus time

    plt.figure(figsize=(8, 6))

    # ensemble average
    values = np.mean(flucts_mag, axis=1)
    errors = np.std(flucts_mag, axis=1, ddof=1) / np.sqrt(sysnum)

    plt.errorbar(bs, values, errors, **ebar_kw)

    plt.title("Fluctuations in magnetisation versus temperature")
    plt.xlabel("$\\beta$")
    plt.ylabel("$\\langle \\sigma_M \\rangle_e$")

    plt.savefig(resultspath / "flucts.pdf")
    plt.show()
    plt.close()

