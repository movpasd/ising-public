"""Script to generate the data while I sleep zz"""

import numpy as np
from pathlib import Path

from ising import simulator, plotter, thermo, datagen, loadingbar


def gen_relaxation(chunknum, chunksize, dset_select="aligned"):
    """
    Simulate chunknum*chunksize steps for the aligned dataset

    chunksize is just passed as iternum to the ensemble.simulate()
    and after each chunk the data is saved.
    """

    datapath = Path(__file__).parents[0] / "data/relaxation"
    dataset = datagen.DataSet(datapath / f"init_{dset_select}")
    dataset.load()

    print(f"Generating {chunknum} chunks of size {chunksize}")
    print(f"for dataset '{dset_select}'")

    for n in range(chunknum):

        print()
        print(f"CHUNK #{n}")
        print(f"===============")
        print()

        for k, ens in enumerate(dataset.ensembles):

            b = ens.b
            print(f"Ensemble b={b:.2f}")
            ens.simulate(chunksize, reset=False, verbose=True)
            dataset.save(ens_index=k)


if __name__ == "__main__":

    gen_relaxation(10, 50)