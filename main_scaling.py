"""Investigating finite-sized scaling"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import simulator, plotter, thermo, datagen


datapath = Path(__file__).parents[0] / "data/scaling"
resultspath = Path(__file__).parents[0] / "results/scaling"


def find_heat_capacity(N, T, tol, sysnum=20):
    """Find heat capacity to specified tolerance"""

    # Basic working:
    # Create ensemble with given parameters.
    # Keep simulating, periodically calculating heat capacity using f-d
    # Error is estimated by avging over ensemble.
    # As soon as the heat capacity is found to a suitable tolerance,
    # return that value.

    relaxtime = 150
    maxtime = 5000
    checktime = 50
    sysnum = 100

    print(f"Finding heat capacity, N={N}, T={T:.2f}, {sysnum} systems\n")

    b = 1 / T
    ensemble = datagen.Ensemble(N, sysnum, p=1, b=b, h=0, randflip=True)

    # initial simulation to reach equilibrium
    ensemble.simulate(relaxtime + 10)
    ensemble.trim_init(relaxtime)

    total_iterations = 10

    done = False

    while total_iterations < maxtime and not done:

        print(end=f"Simulating {total_iterations} -> "
              f"{total_iterations + checktime}: ")

        ensemble.simulate(checktime, reset=False)

        arr = ensemble.asarray()
        energies = thermo.energy(arr)

        flucts = np.std(energies, ddof=1, axis=0)
        est_fluct = np.mean(flucts)
        err_fluct = np.std(flucts, ddof=1) / np.sqrt(sysnum)

        # Error propagation!

        est_cap = est_fluct**2 * b**2
        err_cap = 2 * err_fluct * est_fluct * b**2

        rel_err = err_cap / est_cap

        print(
            f"|err_caps / cap| = |{err_cap:.3f} / {est_cap:.3f}| = {rel_err:.3f}")

        if rel_err < tol:
            done = True

        total_iterations += checktime

    if not done:
        print(f"Exceeded max time of {maxtime}")

    return est_cap, err_cap


def calculate(Ns, Ts, tol):

    for N in Ns:

        ests = []
        errs = []

        for T in Ts:

            est, err = find_heat_capacity(N, T, tol)
            ests.append(est)
            errs.append(err)

            print()

        np.save(datapath / f"ests-N{N}.npy", np.array(ests))
        np.save(datapath / f"errs-N{N}.npy", np.array(errs))


def results(Ns, Ts):

    plt.figure(figsize=(12, 8))

    # colours
    cs = [(.9, 0, 0), (.8, .7, 0), (0, .8, 0), (0, .7, .7),
          (0, 0, .9), (.7, 0, .7), (.5, .5, .5)]

    for k, N in enumerate(Ns):

        ests = np.load(datapath / f"ests-N{N}.npy")
        errs = np.load(datapath / f"errs-N{N}.npy")

        plt.errorbar(Ts, ests / N**2, errs / N**2,
                     fmt="s", markersize=1, color=cs[k],
                     ecolor=cs[k] + (0.3,), elinewidth=3)

    T_ons = 2 / np.log(1 + np.sqrt(2))
    plt.plot([T_ons, T_ons], [0, 1.25], "k--")

    plt.legend(["$T_{ons}$"] + [f"N = {N}" for N in Ns])

    plt.title("Specific heat capacity versus temperature for various N\n"
              "Calculated via F.-D. theorem")
    plt.xlabel("Temperature")
    plt.ylabel("Specific heat capacity $C/N^2$")

    plt.savefig(resultspath / "caps.pdf")
    plt.show()
    plt.close()


ranges = [
    (1, 2, 0.2),
    (2, 4, 0.025),
    (4, 5.1, 0.1)
]

Ns = [2, 3, 4, 5, 6, 7, 8]
Ts = np.concatenate([np.arange(*r) for r in ranges])

# calculate(Ns, Ts, tol=0.05)

# results(Ns, Ts)

Tcs = [2.6, 2.47, 2.45, 2.41, 2.4, 2.37, 2.36]
errs = [.1, .08, .07, .06, .06, .03, .04]

plt.errorbar(Ns, Tcs, errs,
    fmt="x", color="k", ecolor="r", elinewidth=2)

plt.xlabel("N")
plt.ylabel("$T_c$")

plt.savefig(resultspath / "Tcs.pdf")
plt.show()