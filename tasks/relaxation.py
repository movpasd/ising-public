"""Measure relaxation times"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import datagen, loadingbar, plotter, simulator, thermo


def generate(datapath, grid_size=30, sysnum=100, maxiternum=500):
    """Generate the data"""

    print("Generating data\n")

    datapath = Path(datapath)

    init_aligned_dataset = datagen.DataSet(datapath / "init_aligned")
    init_random_dataset = datagen.DataSet(datapath / "init_random")

    # Shared parameters
    h = 0

    # Generate initially aligned simulations
    p = 1.0

    for k in range(11):

        b = 0.1 * k
        print(f"aligned b = {b}")
        ens = datagen.Ensemble(grid_size, sysnum, p, b, h, identical=False)
        ens.simulate(maxiternum, reset=False, verbose=True)
        init_aligned_dataset.add_ensemble(ens, save=True)

    # Generate initially randomised simulations
    p = 0.5

    for k in range(11):

        b = 0.1 * k
        print(f"random b = {b}")
        ens = datagen.Ensemble(grid_size, sysnum, p, b, h, identical=False)
        ens.simulate(maxiternum, reset=False, verbose=True)
        init_random_dataset.add_ensemble(ens, save=True)

    print("Finally saving...")
    init_aligned_dataset.save()
    init_random_dataset.save()
    print("Done generating!\n")


def display_mosaic(datapath, dataset_select, ensemble_select):
    """
    Display the generated ensembles as a mosaic
    (used for testing)
    """

    datapath = Path(datapath)

    if dataset_select == "aligned":
        dataset = datagen.DataSet(datapath / "init_aligned")
    elif dataset_select == "random":
        dataset = datagen.DataSet(datapath / "init_random")
    else:
        raise ValueError(f"what is {dataset_select}?")

    dataset.load()

    if type(ensemble_select) is not int:
        raise ValueError("bad input")
    if ensemble_select < 0 or ensemble_select > 10:
        raise ValueError("please input an int from 0 to 10 inclusive")

    ens = dataset.ensembles[ensemble_select]
    plotter.animate_mosaic(ens, show=True)


def analyse(datapath, rolavg_window=100):
    """Analyse and return magnetisation data"""

    datapath = Path(datapath)

    datasets = [datagen.DataSet(datapath / "init_aligned"),
                datagen.DataSet(datapath / "init_random")]

    datasets[0].load()
    datasets[1].load()

    # Square magnetisation data, indexed by:
    # [dataset, ensemble][time]
    # dataset -- 0: aligned, 1: random
    mags = np.array([
        np.array([
            # ensemble.asarray(): [time, system, Nx, Ny]
            # -> magnetisation(...): [time, system]
            # -> np.mean(..., axis=-1): [time]
            np.mean(thermo.magnetisation(ensemble.asarray()), axis=-1)
            for ensemble in dataset.ensembles
        ])
        for dataset in datasets
    ])

    sqmags = mags**2
    diffs = np.diff(sqmags, axis=-1)
    smoothed_diffs = thermo.rolling_average(diffs, rolavg_window, axis=-1)

    return mags, sqmags, diffs, smoothed_diffs


def graph(resultspath, *args, save=True, show=False):
    """Takes output from analyse() and spits out some pretty graphs"""

    path = Path(resultspath)

    filenames = ["sqmags", "diffs", "smoothed_diffs"]

    for arg, filename in zip(args, filenames):

        for init_condition, index in zip(["aligned", "random"], [0, 1]):

            plt.figure(figsize=(8, 6))

            c = 0
            dc = 1 / 11

            for values in arg[index]:
                plt.plot(values, color=(1 - c, 0, c))
                c += dc

            # plt.title(f"{filename}, initially {init_condition}")
            plt.legend([f"{k * 0.1: .2f}" for k in range(11)])

            plt.xlabel("Time")
            plt.ylabel("Rolling average of $d(M^2)/dt$")

            if save:
                plt.savefig(path / f"{init_condition}-{filename}.pdf")

            if show:
                plt.show()

            plt.close()