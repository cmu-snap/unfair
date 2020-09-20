#!/usr/bin/env python3
""" Visualizes sklearn training parameters. """

import argparse
import itertools
import multiprocessing
import os
from os import path
import random
import shutil
import time

import numpy as np
from matplotlib import pyplot as plt

import cl_args
import defaults
import models
import train
import utils


# Key to use in the results file.
RESULTS_KEY = "results"
# Experiment parameters.
#
# Number of simulations.
NUMS_SIMS = [1, 10, 100, 1000]
# NUMS_SIMS = [1, 10, 100, 1000, 10000]
# Percent of each simulation.
PRC_MIN = 5
PRC_MAX = 50
PRC_DELTA = 5
# PRC_MIN = 10
# PRC_MAX = 100
# PRC_DELTA = 10
# KEEP_PRCS = [100]
KEEP_PRCS = [5, 10, 25]
# Number of iterations.
NUM_ITERS_MIN = 5
NUM_ITERS_MAX = 80
NUM_ITERS_DELTA = 15
# NUM_ITERS_MIN = 5
# NUM_ITERS_MAX = 150
# NUM_ITERS_DELTA = 15
NUMS_ITERS = [5, 20, 80]
# NUMS_ITERS = [5, 20, 65, 155]


def get_partial_results_flp(cnf):
    """ Assembles the results filepath for a configuration. """
    return path.join(
        cnf["out_dir"],
        (f"{cnf['num_sims']}_{cnf['keep_percent']}_"
         f"{cnf['max_iter']}_results.npz"))


def maybe_run_cnf(cnf, func):
    """
    Runs a function on a configuration, unless results already exist
    or a lock file indicates that training is already running in
    another process. Creates a lock file if the function is executed.
    """
    # If results already exist for this configuration, then return them.
    res_flp = get_partial_results_flp(cnf)
    if path.exists(res_flp):
        print(f"Results already exist: {res_flp}")
        with np.load(res_flp) as fil:
            return fil[RESULTS_KEY]
    # If a lock file exists for this configuration, then skip it.
    out_dir = cnf["out_dir"]
    if utils.check_lock_file(out_dir):
        print(f"Configuration running somewhere else: {out_dir}")
        return None
    # Otherwise, create a lock file and process the configuration.
    utils.create_lock_file(out_dir)
    return func(cnf)


def cleanup_combine_and_save_results(cnf, res):
    """
    Extracts the exercised hyper-parameters from cnf, packages them
    with the results, and saves them to disk in cnf's output
    directory. Removes a lock file if one was created.
    """
    # If the result is None, then the training never ran. This occurs when
    # another process is already running training, as evidenced by a lock file.
    if res is None:
        # Training produced no results. The lock file belongs to
        # another process, so do not remove it.
        res = float("NaN"), float("NaN")
    else:
        # Training did, in fact, produce results, so remove the lock file.
        utils.remove_lock_file(cnf["out_dir"])
    # If the result is not already bundled with its parameters, then
    # do so and save it to disk.
    if len(res) != 5:
        los_tst, tim_trn_s = res
        res = np.array(
            [cnf["num_sims"], cnf["keep_percent"], cnf["max_iter"], los_tst,
             tim_trn_s])
        np.savez_compressed(get_partial_results_flp(cnf), **{RESULTS_KEY: res})
    return res


def graph(lines, flp, lbl_leg, lbl_x, lbl_y, lim_y=None):
    """ Generate a single graph. """
    for key, line in sorted(lines.items()):
        xs, ys = zip(*line)
        plt.plot(xs, ys, label=f"{key}{lbl_leg}")
    plt.legend()
    plt.xlabel(lbl_x)
    plt.ylabel(lbl_y)
    if lim_y is not None:
        plt.ylim(lim_y)
    plt.tight_layout()
    print(f"Saving graph: {flp}")
    plt.savefig(flp)
    plt.close()


def graph_partial_results(cnfs, out_dir):
    """ Look for and graph partial results. """
    results = []
    for cnf in cnfs:
        cnf_out_dir = cnf["out_dir"]
        results_flps = [
            path.join(cnf_out_dir, fln) for fln in os.listdir(cnf_out_dir)
            if fln.endswith("_results.npz")]
        assert len(results_flps) <= 1
        if results_flps:
            results_flp = results_flps[0]
            print(f"Loading results: {results_flp}")
            with np.load(results_flp) as fil:
                results.append(fil[RESULTS_KEY])
    len_before = len(results)
    for result in results:
        print(result.tolist())

    results = [res for res in results if not np.isnan(res).any()]
    len_after = len(results)
    print(
        (f"Found {len_after} valid results! (Dropped "
         f"{len_before - len_after} containing NaNs.)"))
    graph_results(results, out_dir)


def graph_results(results, out_dir):
    """ Graph the provided results. """
    # Vary keep_percent. One graph for each number of iterations. A line for
    # each number of simulations.
    #
    # Create one graph for each number of iterations.
    for max_iters_graph in NUMS_ITERS:
        # Create a line for each number of simulations.
        num_sims_lines = {}
        for num_sims_target in NUMS_SIMS:
            num_sims_lines[num_sims_target] = []
            for result in results:
                num_sims, keep_prc, max_iter, los_tst, tim_trn_s = result
                if num_sims == num_sims_target and max_iter == max_iters_graph:
                    num_sims_lines[num_sims_target].append(
                        (keep_prc, los_tst, tim_trn_s))
        graph(
            lines={
                key: [(x, 1 - y) for x, y, _ in val]
                for key, val in num_sims_lines.items()},
            flp=path.join(
                out_dir,
                f"max_iters_{max_iters_graph}-vary_keep_percent-accuracy.pdf"),
            lbl_leg=" simulations",
            lbl_x="Percent kept from each sim",
            lbl_y="Accuracy",
            lim_y=(0, 1))
        graph(
            lines={
                key: [(x, y) for x, _, y in val]
                for key, val in num_sims_lines.items()},
            flp=path.join(
                out_dir,
                f"max_iters_{max_iters_graph}-vary_keep_percent-time.pdf"),
            lbl_leg=" simulations",
            lbl_x="Percent kept from each sim",
            lbl_y="Training time (seconds)")

    # Vary max_iter. One graph for each number of iterations. A line for
    # each number of simulations.
    #
    # Create one graph for each number of iterations.
    for num_sims_graph in NUMS_SIMS:
        # Create a line for each number of simulations.
        keep_prc_lines = {}
        for keep_prc_target in KEEP_PRCS:
            keep_prc_lines[keep_prc_target] = []
            for result in results:
                num_sims, keep_prc, max_iter, los_tst, tim_trn_s = result
                if num_sims == num_sims_graph and keep_prc == keep_prc_target:
                    keep_prc_lines[keep_prc_target].append(
                        (max_iter, los_tst, tim_trn_s))

        graph(
            lines={
                key: [(x, 1 - y) for x, y, _ in val]
                for key, val in keep_prc_lines.items()},
            flp=path.join(
                out_dir,
                f"num_sims_{num_sims_graph}-vary_max_iter-accuracy.pdf"),
            lbl_leg="% kept",
            lbl_x="Number of training iterations",
            lbl_y="Accuracy",
            lim_y=(0, 1))
        graph(
            lines={
                key: [(x, y) for x, _, y in val]
                for key, val in keep_prc_lines.items()},
            flp=path.join(
                out_dir,
                f"num_sims_{num_sims_graph}-vary_max_iter-time.pdf"),
            lbl_leg="% kept",
            lbl_x="Number of training iterations",
            lbl_y="Training time (seconds)")


def main():
    """ This program's entrypoint. """
    psr = argparse.ArgumentParser(
        description="Visualize sklearn training parameters.")
    psr, psr_verify = cl_args.add_training(psr)
    psr.add_argument(
        "--graph-results", action="store_true",
        help=("Look through the output directory for completed experiments, "
              "and graph them."))
    args = psr_verify(psr.parse_args())
    args = train.prepare_args(vars(args))
    tim_srt_s = time.time()
    out_dir = args["out_dir"]

    # Assemble the configurations to test. Sort them based on the
    # product of their hyper-parameters, which is a heuristic of how
    # long they will take to run.
    cnfs = sorted(
        [train.prepare_args({
            "warmup_percent": args["warmup_percent"],
            "model": args["model"],
            "kernel": args["kernel"],
            "degree": args["degree"],
            "penalty": args["penalty"],
            "standardize": args["standardize"],
            "no_rand": args["no_rand"],
            "max_iter": max_iter,
            "keep_percent": prc,
            "num_sims": num_sims,
            "out_dir": path.join(out_dir, f"{num_sims}_{prc}_{max_iter}")
        }) for num_sims, prc, max_iter in set(
            # Fix number of iterations and number of
            # simulations. Vary percent of each simulation.
            list(itertools.product(
                NUMS_SIMS, range(PRC_MIN, PRC_MAX + 1, PRC_DELTA),
                NUMS_ITERS)) +
            # Fix percent of each simulation and number of
            # simulations. Vary number of iterations.
            list(itertools.product(
                NUMS_SIMS, KEEP_PRCS,
                range(NUM_ITERS_MIN, NUM_ITERS_MAX + 1, NUM_ITERS_DELTA))))],
        key=lambda cnf: np.prod(
            [cnf["num_sims"], cnf["keep_percent"], cnf["max_iter"]]))
    print(f"Will test {len(cnfs)} configurations.")

    if args["graph_results"]:
        graph_partial_results(cnfs, out_dir)
        return

    # For each possible configuration of number of simulations and
    # percentage of each simulation, create a temporary file
    # containing the parsed data for that configuration.
    all_prcs = list(set(
        KEEP_PRCS + list(range(PRC_MIN, PRC_MAX + 1, PRC_DELTA))))
    tmp_dat = {}
    for num_sims, prc in itertools.product(NUMS_SIMS, all_prcs):
        base_dir = path.join(out_dir, f"{num_sims}_{prc}")
        if not path.exists(base_dir):
            os.makedirs(base_dir)
        tmp_dat_flp = path.join(base_dir, "data.npz")
        tmp_scl_prms_flp = path.join(base_dir, "scale_params.json")
        # Record the paths to the data and scale parameters files.
        tmp_dat[(num_sims, prc)] = (tmp_dat_flp, tmp_scl_prms_flp)

    # Look up to the location of the data corresponding to each
    # configuration and make a mapping from this temporary data to
    # where it should be copied in the configuration's output
    # directory (where it will be read automatically).
    src_dst = []
    for cnf in cnfs:
        cnf_out_dir = cnf["out_dir"]
        if not path.exists(cnf_out_dir):
            os.makedirs(cnf_out_dir)
        tmp_dat_flp, tmp_scl_prms_flp = tmp_dat[
            (cnf["num_sims"], cnf["keep_percent"])]
        src_dst.append((
            (tmp_dat_flp,
             path.join(cnf_out_dir, path.basename(tmp_dat_flp))),
            (tmp_scl_prms_flp,
             path.join(cnf_out_dir, path.basename(tmp_scl_prms_flp)))))

    # Check if any of the data has not been generated yet. If any of
    # the data has not been generated yet, then we must regenerate all
    # of the data.
    if np.array([
            path.exists(dat_dst) and path.exists(scl_prms_dst)
            for (_, dat_dst), (_, scl_prms_dst) in src_dst]).all():
        print("All data already generated.")
    else:
        print("Generating all new data.")
        # The data processing here is tricky. Our goal is to have
        # the configurations all process the same data to make the
        # comparison as "apples-to-apples" as possible. To that
        # end, all of the configurations will process subsets of
        # the same set of simulations. The high-level strategy is
        # to manually load and process the data and then place it
        # strategically so that the training process reads it in
        # automatically. The process is as follows:
        #    1) Determine the greatest number of simulations that any
        #       configuration will require and parse them.
        #    2) For each configuration, pick a subset of these simulations.
        #       When picking which simulations to use, always pick from the
        #       front. This means that configurations with more simulations
        #       will process the same simulations as those with fewer
        #       simulations, plus some extra. When picking which datapoints
        #       to use from within a simulation, choose them randomly.
        #    3) For each configuration, copy its data into its output
        #       directory so that it will be automatically detected and read
        #       in.

        # Load the required number of simulations.
        dat_dir = args["data_dir"]
        sims = [path.join(dat_dir, sim) for sim in os.listdir(dat_dir)]
        if train.SHUFFLE:
            # Set the random seed so that multiple instances of this
            # script see the same random order.
            utils.set_rand_seed()
            random.shuffle(sims)
        num_sims_actual = len(sims)
        max_sims = max(NUMS_SIMS)
        assert num_sims_actual >= max_sims, \
            (f"Insufficient simulations. Requested {max_sims}, but only "
             f"{num_sims_actual} available.")
        sims = sims[:max_sims]
        net = models.MODELS[args["model"]]()
        sim_args = [
            (idx, max_sims, net, sim_flp, out_dir, args["warmup_percent"], 100)
            for idx, sim_flp in enumerate(sims)]
        if defaults.SYNC:
            dat_all = [train.process_sim(*sim_args_) for sim_args_ in sim_args]
        else:
            with multiprocessing.Pool() as pol:
                dat_all = pol.starmap(train.process_sim, sim_args)
        # Verify that we were able to parse all
        # simulations. Normally, we would allow the training
        # process to proceed even if some simulations failed to
        # parse, but since in this case we are looking at specific
        # trends, we need to make sure that we are training on the
        # number of simulations that we intend.
        for dat in dat_all:
            assert dat is not None, \
                "Error processing at least one simulation. Check logs (above)."
        # Unpack the data.
        dat_all, sims = zip(*dat_all)
        dat_all = [utils.load_tmp_file(flp) for flp in dat_all]
        dat_in, dat_out, dat_extra, scl_grps = zip(*dat_all)
        dat_in = list(dat_in)
        dat_out = list(dat_out)
        dat_extra = list(dat_extra)
        scl_grps = list(scl_grps)

        # Generate temporary data.
        for (num_sims, prc), (tmp_dat_flp, tmp_scl_prms_flp) in tmp_dat.items():
            # Select the data corresponding to this number of
            # simulations and percent of each simulation.
            dat_all = list(zip(*utils.filt(
                dat_in, dat_out, dat_extra, scl_grps, num_sims, prc)))
            # Finish processesing the data and save it in a form that
            # can be read by the training process.
            ignore = train.gen_data(
                net, args, dat_flp=tmp_dat_flp, scl_prms_flp=tmp_scl_prms_flp,
                dat=(dat_all, sims), save_data=True)
            del ignore

        # Copy temporary data to configuration output directories.
        for (dat_src, dat_dst), (scl_prms_src, scl_prms_dst)  in src_dst:
            shutil.copyfile(dat_src, dat_dst)
            shutil.copyfile(scl_prms_src, scl_prms_dst)

        # Remove temporary data files.
        for num_sims, prc in itertools.product(NUMS_SIMS, all_prcs):
            tmp_dir = path.join(out_dir, f"{num_sims}_{prc}")
            print(f"Removing: {tmp_dir}")
            shutil.rmtree(tmp_dir)

    # Train models.
    train.run_cnfs(
        cnfs, defaults.SYNC, maybe_run_cnf, cleanup_combine_and_save_results)

    # # Remove real data files.
    # for cnf in cnfs:
    #     print(f"Removing: {cnf['out_dir']}")
    #     shutil.rmtree(cnf["out_dir"])

    graph_partial_results(cnfs, out_dir)
    print(f"Total time: {time.time() - tim_srt_s:.2f} seconds")
    return


if __name__ == "__main__":
    main()
