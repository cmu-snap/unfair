#! /usr/bin/env python3
""" Plot histograms of the training features. """

import argparse
import os
from os import path

from matplotlib import pyplot
import numpy as np


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(
        description="Visualize a simulation's features.")
    psr.add_argument(
        "--training-data", help="The path to the parsed training data.",
        required=True, type=str)
    psr.add_argument(
        "--out-dir", default=".",
        help="The directory in which to store output files.", type=str)
    args = psr.parse_args()
    dat_flp = args.training_data
    out_dir = args.out_dir
    assert path.exists(dat_flp), f"File does not exist: {dat_flp}"
    if not path.exists(out_dir):
        os.makedirs(out_dir)

    # Read data.
    dat = np.load(dat_flp)
    num_arrays = len(dat.files)
    assert num_arrays == 5, f"Expected 5 arrays, but found: {dat.files}"
    dat_in = dat["dat_in"]

    # Generate graphs.
    for fet in dat_in.dtype.names:
        print(f"Plotting feature: {fet}")
        pyplot.hist(dat_in[fet], bins=50, density=True)
        pyplot.xlabel(fet)
        pyplot.ylabel("histogram")
        pyplot.savefig(path.join(
            out_dir, f"{fet.replace(' ', '_').replace('/', '-')}.pdf"))
        pyplot.close()


if __name__ == "__main__":
    main()
