"""Investigating hysteresis effects"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import datagen, thermo, plotter


hparams = {
    "maxh": 1,
    "period": 200
}

relaxtime = 200


datapath = Path(__file__).parents[0] / "data/hysteresis"
resultspath = Path(__file__).parents[0] / "results/hysteresis"


def get_hysteresis_loop(b, N, iternum, sysnum, hparams, anim=False):
    """
    Compute and return magnetisation v time values

    RETURNS: mags -- (iternum, sysnum)-array 
             hs -- (iternum,)-array
    """

    a, p = hparams["maxh"], hparams["period"]
    hs = a * np.sin(2 * np.pi * np.arange(0, p) / p)

    # In this case, domain walls probably actually lead to
    # interesting behaviour, so set p = 0.5

    # Initially the applied field is zero, but we'll turn it on
    # once the relaxation time has elapsed
    ensemble = datagen.Ensemble(N, sysnum, p=0.5, b=b, h=np.zeros(1))

    ensemble.simulate(relaxtime + 1)
    ensemble.trim_init(relaxtime)

    ensemble.hs = hs
    ensemble.simulate(iternum - 1, reset=False, verbose=True)

    if anim:

        print("animating:")
        fig, _, _ = plotter.animate_mosaic(
            ensemble, timestamp=True,
            saveas=resultspath / f"N{N}-b{b:.2f}.mp4", verbose=True
        )
        plt.close(fig)

    arr = ensemble.asarray()
    mags = thermo.magnetisation(arr)

    return np.resize(hs, iternum), mags


def generate(bs, N, iternum, sysnum):

    print("Generating")

    for b in bs:

        print(f"generate b: {b}")

        hs, mags = get_hysteresis_loop(b, N, iternum, sysnum, hparams,
                                       anim=True)

        np.save(datapath / f"mags-N{N}-b{b:.2f}.npy", mags)
        np.save(datapath / f"hs-N{N}-b{b:.2f}.npy", hs)


def results(bs, N):

    print("Drawing results")

    for b in bs:

        print(f"plot b: {b}")

        mags = np.load(datapath / f"mags-N{N}-b{b:.2f}.npy")
        hs = np.load(datapath / f"hs-N{N}-b{b:.2f}.npy")

        iternum, sysnum = mags.shape

        plt.figure(figsize=(6, 6))

        est_mags = np.mean(mags, axis=1)
        err_mags = np.std(mags, axis=1, ddof=1) / np.sqrt(sysnum)

        hmax = hparams["maxh"]
        mmax = np.max(est_mags)
        plt.plot([-hmax, +hmax], [0, 0], "-", color="grey", lw=0.5)
        plt.plot([0, 0], [-mmax, mmax], "-", color="grey", lw=0.5)

        plt.errorbar(hs, est_mags, err_mags,
                     fmt="k+", ms=4, markeredgewidth=0.5,
                     ecolor=(1, 0, 0, 0.3), elinewidth=2)
        # plt.plot(hs, est_mags, "k:", linewidth=1)

        plt.title(f"Hysteresis effect with oscillating applied field\n"
            f"$\\beta$ = {b:.2f}, N = {N}, period of {hparams['period']}")
        plt.xlabel("$H$")
        plt.ylabel("$\\langle M \\rangle_e$")

        plt.savefig(resultspath / f"loop-N{N}-b{b:.2f}.pdf")
        plt.show()
        plt.close()


bs = np.arange(0, 1.1, 0.1)
N = 20
iternum = 2 * hparams["period"]
sysnum = 16

generate(bs, N, iternum, sysnum)
results(bs, N)