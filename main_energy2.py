"""
Attempt at studying heat capacity across different Ns

I thought it would be easier to just copy paste the code from main_energy
rather than modify it to work for multiple Ns.

In the end opted for a F.d. solution -- see main_scaling.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import datagen, thermo

datapath = Path(__file__).parents[0] / "data/energy/varN"
autocdatapath = Path(__file__).parents[0] / "data/autoc-new"
resultspath = Path(__file__).parents[0] / "results/energy"

# ensembles will be sorted into lists depending on N by load_ensembles()
Ns = []
ensdict = {}


def load_ensembles():

    print("Loading data")

    # Unpacking the autoc data is a bit more complicated than in main_energy.py
    # because we have to sort the data by N _and_ by b.

    dataset = datagen.DataSet(autocdatapath, load=True)

    for ens in dataset.ensembles:

        N, b = ens.grid_shape[0], ens.b

        if N not in Ns:

            ensdict[N] = []
            Ns.append(N)

        ensdict[N].append(ens)

    np.save(datapath / "Ns.npy", np.array(Ns))


def calculate():

    load_ensembles()

    for i, N in enumerate(Ns):

        # This is just the code from main_energy.py but for this particular
        # value of N, indexed by i. See that file for comments

        print(f"Calculating N{i} = {N}")

        energies, flucts, bs = [], [], []

        for k, ens in enumerate(ensdict[N]):

            print(end=".")
            ener_arr = thermo.energy(ens.asarray())
            energies.append(np.mean(ener_arr, axis=0))
            flucts.append(np.std(ener_arr, axis=0))
            bs.append(ens.b)

        bs = np.array(bs)
        energies = np.stack(energies, axis=0)
        flucts = np.stack(flucts, axis=0)
        np.save(datapath / f"bs-N{i}.npy", bs)
        np.save(datapath / f"energies-N{i}.npy", energies)
        np.save(datapath / f"flucts-N{i}.npy", flucts)

        print()

        # bs = np.load(datapath / f"bs-N{i}.npy")
        # energies = np.load(datapath / f"energies-N{i}.npy")
        # flucts = np.load(datapath / f"flucts-N{i}.npy")

        sysnum = energies.shape[1]

        est_energies = np.mean(energies, axis=1)
        err_energies = np.std(energies, axis=1) / np.sqrt(sysnum)

        midbs = (bs[1:] + bs[:-1]) / 2
        Ts = 1 / midbs

        caps = -midbs**2 * np.diff(est_energies) / np.diff(bs)
        err_caps = caps * (np.sqrt(err_energies[1:]**2 + err_energies[:-1]**2)
                           / np.abs(np.diff(est_energies)))

        est_flucts = np.mean(flucts, axis=1)
        err_flucts = np.std(flucts, axis=1) / np.sqrt(sysnum)

        np.save(datapath / f"calculations-N{i}.npy",
                np.stack([Ts, caps, err_caps], axis=0))


def results(iternum=0):

    Ns = np.load(datapath / "Ns.npy")

    plt.figure(figsize=(12, 8))

    onsager_Tc = 2 / np.log(1 + np.sqrt(2))
    cs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0.7, 0.7)]

    for i, N in enumerate(Ns):

        print(f"Plotting N{i} = {N}")

        Ts, caps, err_caps = (
            np.load(datapath / f"calculations-N{i}.npy"))

        plt.errorbar(Ts, caps / N**2, err_caps / N**2,
                     fmt="x", ms=8, color=cs[i], 
                     ecolor=cs[i] + (.5,), elinewidth=1)
        # plt.plot(Ts, caps / N**2, "", color=cs[i])

    plt.legend([f"N={N}" for N in Ns])

    plt.plot([onsager_Tc, onsager_Tc], [0, 1.75], "k--", lw=0.5)

    plt.title("Heat capacity vs temperature for various N")
    plt.xlabel("$T$")
    plt.ylabel("$C / N^2$")

    plt.savefig(resultspath / f"cap_varN.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":

    # calculate()
    results()
