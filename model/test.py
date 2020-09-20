#! /usr/bin/env python3
""" Evalaute the model on simulations. """

import argparse
import json
import multiprocessing
import os
from os import path
import pickle
from statistics import mean
import time
from matplotlib import pyplot

import numpy as np
import torch

import cl_args
import defaults
import models
import train
import utils


def plot_bar(x_axis, y_axis, file_name):
    """ Create a bar graph. """
    y_pos = np.arange(len(y_axis))
    pyplot.bar(y_pos, y_axis, align="center", alpha=0.5)
    pyplot.xticks(y_pos, x_axis)
    pyplot.ylabel("Accuracy")
    pyplot.tight_layout()
    pyplot.savefig(file_name)
    pyplot.close()


def init_global(manager):
    """ Initialize global variables that are shared by all processes. """
    # BW in Mbps.
    bw_dict = manager.dict({
        1: manager.list(),
        10: manager.list(),
        30: manager.list(),
        50: manager.list(),
        1000: manager.list()
    })

    # RTT in us
    rtt_dict = manager.dict({
        1000: manager.list(),
        10000: manager.list(),
        50000: manager.list(),
        100000: manager.list(),
        1000000: manager.list()
    })

    # Queue size in BDP
    queue_dict = manager.dict({
        1: manager.list(),
        2: manager.list(),
        4: manager.list(),
        8: manager.list(),
        16: manager.list(),
        32: manager.list(),
        64: manager.list()
    })

    # BW in Mbps.
    bucketized_bw_dict = manager.dict({
        1: manager.list(),
        10: manager.list(),
        30: manager.list(),
        50: manager.list(),
        1000: manager.list()
    })

    # RTT in us
    bucketized_rtt_dict = manager.dict({
        1000: manager.list(),
        10000: manager.list(),
        50000: manager.list(),
        100000: manager.list(),
        1000000: manager.list()
    })

    # Queue size in BDP
    bucketized_queue_dict = manager.dict({
        1: manager.list(),
        2: manager.list(),
        4: manager.list(),
        8: manager.list(),
        16: manager.list(),
        32: manager.list(),
        64: manager.list()
    })

    return (bw_dict, rtt_dict, queue_dict, bucketized_bw_dict,
            bucketized_rtt_dict, bucketized_queue_dict)


def process_one(idx, total, sim_flp, out_dir, net, warmup_prc, scl_prms_flp,
                standardize, all_accuracy, all_bucketized_accuracy, bw_dict,
                rtt_dict, queue_dict, bucketized_bw_dict, bucketized_rtt_dict,
                bucketized_queue_dict, lock):
    """ Evaluate a single simulation. """
    if not path.exists(out_dir):
        os.makedirs(out_dir)

    # Load and parse the simulation.
    temp_path, sim = (
        train.process_sim(
            idx, total, net=net, sim_flp=sim_flp, tmp_dir=out_dir,
            warmup_prc=warmup_prc, keep_prc=100, sequential=True))

    dat_in, dat_out, dat_extra, _ = utils.load_tmp_file(temp_path)

    # Load and apply the scaling parameters.
    with open(scl_prms_flp, "r") as fil:
        scl_prms = json.load(fil)
    dat_in = utils.scale_all(dat_in, scl_prms, 0, 1, standardize)

    # Visualize the ground truth data.
    utils.visualize_classes(net, dat_out)

    # Test the simulation.
    accuracy, bucketized_accuracy = net.test(
        *utils.Dataset(
            fets=dat_in.dtype.names,
            dat_in=utils.clean(dat_in),
            dat_out=utils.clean(dat_out),
            dat_extra=dat_extra).raw(),
        graph_prms={
            "out_dir": out_dir,
            "sort_by_unfairness": False,
            "dur_s": sim.dur_s
        })

    lock.acquire()
    all_accuracy.append(accuracy)
    mean_accuracy = mean(all_accuracy)

    all_bucketized_accuracy.append(bucketized_accuracy)
    mean_bucketized_accuracy = mean(all_bucketized_accuracy)

    for bw_mbps in bw_dict.keys():
        if sim.bw_Mbps <= bw_mbps:
            bw_dict[bw_mbps].append(accuracy)
            bucketized_bw_dict[bw_mbps].append(bucketized_accuracy)
            break

    rtt_us = (sim.btl_delay_us + 2 * sim.edge_delays[0]) * 2
    for rtt_us_ in rtt_dict.keys():
        if rtt_us <= rtt_us_:
            rtt_dict[rtt_us_].append(accuracy)
            bucketized_rtt_dict[rtt_us_].append(bucketized_accuracy)
            break

    bdp = sim.bw_Mbps * rtt_us / sim.payload_B / sim.queue_p

    for queue_bdp in queue_dict.keys():
        if bdp <= queue_bdp:
            queue_dict[queue_bdp].append(accuracy)
            bucketized_queue_dict[queue_bdp].append(bucketized_accuracy)
            break

    print(
        f"Finish processing {sim.name}\n"
        "----Average accuracy for all the processed simulations: "
        f"{mean_accuracy}\n"
        "----Average bucketized accuracy for all the processed simulations: "
        f"{mean_bucketized_accuracy}\n")

    for bw_mbps in bw_dict.keys():
        if bw_dict[bw_mbps]:
            bw_accuracy = mean(bw_dict[bw_mbps])
            bucketized_bw_accuracy = mean(bucketized_bw_dict[bw_mbps])
            print(
                f"----Bandwidth less than {bw_mbps}Mbps accuracy {bw_accuracy}"
                "\n"
                f"    bucketized accuracy {bucketized_bw_accuracy}")

    for rtt_us_ in rtt_dict.keys():
        if rtt_dict[rtt_us_]:
            rtt_accuracy = mean(rtt_dict[rtt_us_])
            bucketized_rtt_accuracy = mean(bucketized_rtt_dict[rtt_us_])
            print(f"----Rtt less than {rtt_us_}ns accuracy {rtt_accuracy}\n"
                  f"    bucketized accuracy {bucketized_rtt_accuracy}")

    for queue_bdp in queue_dict.keys():
        if queue_dict[queue_bdp]:
            queue_accuracy = mean(queue_dict[queue_bdp])
            bucketized_queue_accuracy = mean(bucketized_queue_dict[queue_bdp])
            print(
                f"----Queue size less than {queue_bdp} BDP accuracy "
                f"{queue_accuracy}\n"
                f"    bucketized accuracy {bucketized_queue_accuracy}")
    lock.release()


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="Analyzes full simulations.")
    psr, psr_verify = cl_args.add_running(psr)
    psr.add_argument(
        "--model", help="The path to a trained model file.", required=True,
        type=str)
    psr.add_argument(
        "--simulations",
        help=("The path to a directory of simulations to analyze, or the path "
              "to a single simulation file."),
        required=False, default=None, type=str)
    psr.add_argument(
        "--input-file",
        help=("The path to a file containing a list of simulations."),
        required=False, default=None, type=str)
    args = psr_verify(psr.parse_args())
    mdl_flp = args.model
    out_dir = args.out_dir
    standardize = args.standardize
    input_file = args.input_file
    sim_dir = args.simulations

    assert (sim_dir is None) != (input_file is None), \
        "Test takes in either a directory of simulations or " \
        "an input file containing all the simulations"
    assert path.exists(mdl_flp), f"Model file does not exist: {mdl_flp}"
    if args.simulations:
        assert path.exists(sim_dir), \
            f"Simulation dir/file does not exist: {sim_dir}"
        sim_flps = (
            [path.join(sim_dir, sim_fln) for sim_fln in os.listdir(sim_dir)]
            if path.isdir(sim_dir) else [sim_dir])
    else:
        with open(input_file, "r") as input_file:
            sim_flps = [line.rstrip("\n") for line in input_file]

    # Parse the model filepath to determine the model type, and instantiate it.
    net = models.MODELS[
        # Convert the model filename to an arguments dictionary, and
        # extract the "model" key.
        utils.str_to_args(
            path.basename(mdl_flp),
            order=sorted(defaults.DEFAULTS.keys())
        )["model"]]()
    # # Manually remove the loss event rate sqrt feature.
    # net.in_spc.remove("loss event rate sqrt")
    # Load the model.
    if mdl_flp.endswith("pickle"):
        with open(mdl_flp, "rb") as fil:
            mdl = pickle.load(fil)
    elif mdl_flp.endswith("pth"):
        mdl = torch.jit.load(mdl_flp)
    else:
        raise Exception(f"Unknown model type: {mdl_flp}")
    net.net = mdl
    net.graph = True

    manager = multiprocessing.Manager()
    all_accuracy = manager.list()
    all_bucketized_accuracy = manager.list()

    (bw_dict, rtt_dict, queue_dict, bucketized_bw_dict,
     bucketized_rtt_dict, bucketized_queue_dict) = init_global(manager)

    lock = manager.Lock()

    total = len(sim_flps)
    func_input = [
        (idx, total, sim_flp,
         path.join(out_dir, path.basename(sim_flp).split(".")[0]), net,
         args.warmup_percent, args.scale_params, standardize, all_accuracy,
         all_bucketized_accuracy, bw_dict, rtt_dict, queue_dict,
         bucketized_bw_dict, bucketized_rtt_dict,
         bucketized_queue_dict, lock)
        for idx, sim_flp in enumerate(sim_flps)]

    print(f"Num files: {len(func_input)}")
    tim_srt_s = time.time()
    with multiprocessing.Pool() as pol:
        pol.starmap(process_one, func_input)

    print(f"Done Processing - time: {time.time() - tim_srt_s:.2f} seconds")

    mean_accuracy = mean(all_accuracy)

    with open(path.join(out_dir, "results.txt"), "w") as fil:
        fil.write(
            "Average accuracy for all the processed simulations: "
            f"{mean_accuracy}\n")

        x_axis = []
        y_axis = []

        for bw_mbps, values in bw_dict.items():
            if values:
                bw_accuracy = mean(values)
                fil.write(
                    f"Bandwidth <= {bw_mbps} Mbps, accuracy "
                    f"{bw_accuracy}\n")

                x_axis.append(f"{bw_mbps}Mbps")
                y_axis.append(bw_accuracy)

        plot_bar(
            x_axis, y_axis, path.join(out_dir, "bandwidth_vs_accuracy.pdf"))

        x_axis.clear()
        y_axis.clear()

        for rtt_us, values in rtt_dict.items():
            if values:
                rtt_accuracy = mean(values)
                fil.write(f"Rtt <= {rtt_us} us, accuracy {rtt_accuracy}\n")

                x_axis.append(f"{rtt_us}us")
                y_axis.append(rtt_accuracy)

        plot_bar(x_axis, y_axis, path.join(out_dir, "rtt_vs_accuracy.pdf"))

        x_axis.clear()
        y_axis.clear()

        for queue_bdp, values in queue_dict.items():
            if values:
                queue_accuracy = mean(values)
                fil.write(
                    f"Queue size <= {queue_bdp}x BDP, accuracy "
                    f"{queue_accuracy}\n")

                x_axis.append(f"{queue_bdp}bdp")
                y_axis.append(queue_accuracy)

        plot_bar(x_axis, y_axis, path.join(out_dir, "queue_vs_accuracy.pdf"))


if __name__ == "__main__":
    main()
