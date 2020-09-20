#!/usr/bin/env python3
"""
Based on:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
"""

import argparse
import copy
import functools
import json
import math
import multiprocessing
import os
from os import path
import pickle
import random
import sys
import time

import numpy as np
from numpy.lib import recfunctions
import torch

import cl_args
import defaults
import models
import utils


# The threshold of the new throughout to the old throughput above which a
# a training example will not be considered. I.e., the throughput must have
# decreased to less than this fraction of the original throughput for a
# training example to be considered.
NEW_TPT_TSH = 0.925
# Set to false to parse the simulations in sorted order.
SHUFFLE = True
# The number of times to log progress during one epoch.
LOGS_PER_EPC = 5
# The number of validation passes per epoch.
VALS_PER_EPC = 15


def scale_fets(dat, scl_grps, standardize=False):
    """
    Returns a copy of dat with the columns normalized. If standardize
    is True, then the scaling groups are normalized to a mean of 0 and
    a variance of 1. If standardize is False, then the scaling groups
    are normalized to the range [0, 1]. Also returns an array of shape
    (dat_all[0].shape[1], 2) where row i contains the scaling
    parameters of column i in dat. If standardize is True, then the
    scaling parameters are the mean and standard deviation of that
    column's scaling group. If standardize is False, then the scaling
    parameters are the min and max of that column's scaling group.
    """
    fets = dat.dtype.names
    assert fets is not None, \
        f"The provided array is not structured. dtype: {dat.dtype.descr}"
    assert len(scl_grps) == len(fets), \
        f"Invalid scaling groups ({scl_grps}) for dtype ({dat.dtype.descr})!"

    # Determine the unique scaling groups.
    scl_grps_unique = set(scl_grps)
    # Create an empty array to hold the min and max values (i.e.,
    # scaling parameters) for each scaling group.
    scl_grps_prms = np.empty((len(scl_grps_unique), 2), dtype="float64")
    # Function to reduce a structured array.
    rdc = (lambda fnc, arr:
           fnc(np.array(
               [fnc(arr[fet]) for fet in arr.dtype.names if fet != ""])))
    # Determine the min and the max of each scaling group.
    for scl_grp in scl_grps_unique:
        # Determine the features in this scaling group.
        scl_grp_fets = [fet for fet_idx, fet in enumerate(fets)
                        if scl_grps[fet_idx] == scl_grp]
        # Extract the columns corresponding to this scaling group.
        fet_values = dat[scl_grp_fets]
        # Record the min and max of these columns.
        scl_grps_prms[scl_grp] = [
            np.mean(utils.clean(fet_values))
            if standardize else rdc(np.min, fet_values),
            np.std(utils.clean(fet_values))
            if standardize else rdc(np.max, fet_values)
        ]

    # Create an empty array to hold the min and max values (i.e.,
    # scaling parameters) for each column (i.e., feature).
    scl_prms = np.empty((len(fets), 2), dtype="float64")
    # Create an empty array to hold the rescaled features.
    new = np.empty(dat.shape, dtype=dat.dtype)
    # Rescale each feature based on its scaling group's min and max.
    for fet_idx, fet in enumerate(fets):
        # Look up the parameters for this feature's scaling group.
        prm_1, prm_2 = scl_grps_prms[scl_grps[fet_idx]]
        # Store this min and max in the list of per-column scaling parameters.
        scl_prms[fet_idx] = np.array([prm_1, prm_2])
        fet_values = dat[fet]
        if standardize:
            # prm_1 is the mean and prm_2 is the standard deviation.
            scaled = (
                # Handle the rare case where the standard deviation is
                # 0 (meaning that all of the feature values are the
                # same), in which case return an array of zeros.
                np.zeros(
                    fet_values.shape, dtype=fet_values.dtype) if prm_2 == 0
                else (fet_values - prm_1) / prm_2)
        else:
            # prm_1 is the min and prm_2 is the max.
            scaled = (
                # Handle the rare case where the min and the max are
                # the same (meaning that all of the feature values are
                # the same.
                np.zeros(
                    fet_values.shape, dtype=fet_values.dtype) if prm_1 == prm_2
                else utils.scale(
                    fet_values, prm_1, prm_2, min_out=0, max_out=1))
        new[fet] = scaled

    return new, scl_prms


def process_sim(idx, total, net, sim_flp, tmp_dir, warmup_prc, keep_prc,
                sequential=False):
    """
    Loads and processes data from a single simulation.

    For logging purposes, "idx" is the index of this simulation amongst "total"
    simulations total. Uses "net" to determine the relevant input and output
    features. "sim_flp" is the path to the simulation file. The parsed results
    are stored in "tmp_dir". Drops the first "warmup_prc" percent of packets.
    Of the remaining packets, only "keep_prc" percent are kept. See
    utils.save_tmp_file() for the format of the results file.

    Returns the path to the results file and a descriptive utils.Sim object.
    """
    sim, dat = utils.load_sim(
        sim_flp, msg=f"{idx + 1:{f'0{len(str(total))}'}}/{total}")
    if dat is None:
        return None

    # Drop the first few packets so that we consider steady-state behavior only.
    dat = dat[math.floor(dat.shape[0] * warmup_prc / 100):]
    # Split each data matrix into two separate matrices: one with the input
    # features only and one with the output features only. The names of the
    # columns correspond to the feature names in in_spc and out_spc.
    assert net.in_spc, f"{net.name}: Empty in spec."
    num_out_fets = len(net.out_spc)
    # This is not a strict requirement from a modeling point of view,
    # but is assumed to make data processing easier.
    assert num_out_fets == 1, \
        (f"{net.name}: Out spec must contain a single feature, but actually "
         f"contains: {net.out_spc}")
    dat_in = recfunctions.repack_fields(dat[net.in_spc])
    dat_out = recfunctions.repack_fields(dat[net.out_spc])
    # Create a structured array to hold extra data that will not be
    # used as features but may be needed by the training/testing
    # process.
    raw_dtype = [typ for typ in dat.dtype.descr if typ[0] in net.out_spc][0][1]
    dtype = ([("raw", raw_dtype), ("num_flws", "int32"),
              ("btk_throughput", "int32")] +
             [typ for typ in dat.dtype.descr if typ[0] in defaults.EXTRA_FETS])
    dat_extra = np.empty(shape=dat.shape, dtype=dtype)
    dat_extra["raw"] = dat_out
    dat_extra["num_flws"].fill(sim.unfair_flws + sim.fair_flws)
    dat_extra["btk_throughput"].fill(sim.bw_Mbps)
    for typ in defaults.EXTRA_FETS:
        dat_extra[typ] = dat[typ]
    dat_extra = recfunctions.repack_fields(dat_extra)

    # Convert output features to class labels.
    dat_out = net.convert_to_class(sim, dat_out)

    # If the results contains NaNs or Infs, then discard this
    # simulation.
    def has_non_finite(arr):
        for fet in arr.dtype.names:
            if not np.isfinite(arr[fet]).all():
                print(
                    f"    Simulation {sim_flp} has NaNs of Infs in feature "
                    f"{fet}")
                return True
        return False
    if has_non_finite(dat_in) or has_non_finite(dat_out):
        return None

    # Verify data.
    assert dat_in.shape[0] == dat_out.shape[0], \
        f"{sim_flp}: Input and output should have the same number of rows."
    # Find the uniques classes in the output features and make sure
    # that they are properly formed. Assumes that dat_out is a
    # structured numpy array containing a column named "class".
    for cls in set(dat_out["class"].tolist()):
        assert 0 <= cls < net.num_clss, f"Invalid class: {cls}"

    # Transform the data as required by this specific model.
    dat_in, dat_out, dat_extra, scl_grps = net.modify_data(
        sim, dat_in, dat_out, dat_extra, sequential=sequential)

    # Select a fraction of the data.
    if keep_prc != 100:
        num_rows = dat_in.shape[0]
        num_to_pick = math.ceil(num_rows * keep_prc / 100)
        idxs = np.random.random_integers(0, num_rows - 1, num_to_pick)
        dat_in = dat_in[idxs]
        dat_out = dat_out[idxs]
        dat_extra = dat_extra[idxs]

    # To avoid errors with sending large matrices between processes,
    # store the results in a temporary file.
    dat_flp = path.join(tmp_dir, f"{path.basename(sim_flp)[:-4]}_tmp.npz")
    utils.save_tmp_file(
        dat_flp, dat_in, dat_out, dat_extra, scl_grps)
    return dat_flp, sim


def make_datasets(net, args, dat=None):
    """
    Parses the simulation files in data_dir and transforms them (e.g., by
    scaling) into the correct format for the network.

    If num_sims is not None, then this function selects the first num_sims
    simulations only. If shuffle is True, then the simulations will be parsed in
    sorted order. Use num_sims and shuffle=True together to simplify debugging.
    """
    if dat is None:
        # Find simulations.
        sims = args["sims"]
        if not sims:
            dat_dir = args["data_dir"]
            sims = [
                path.join(dat_dir, sim) for sim in sorted(os.listdir(dat_dir))]
        if SHUFFLE:
            # Set the random seed so that multiple parallel instances of
            # this script see the same random order.
            utils.set_rand_seed()
            random.shuffle(sims)
        num_sims = args["num_sims"]
        if num_sims is not None:
            num_sims_actual = len(sims)
            assert num_sims_actual >= num_sims, \
                (f"Insufficient simulations. Requested {num_sims}, but only "
                 f"{num_sims_actual} available.")
            sims = sims[:num_sims]
        tot_sims = len(sims)
        print(f"Found {tot_sims} simulations.")

        # Prepare temporary output directory. The output of parsing each
        # simulation it written to disk instead of being transfered
        # between processes because sometimes the data is too large for
        # Python to send between processes.
        tmp_dir = args["tmp_dir"]
        if tmp_dir is None:
            tmp_dir = args["out_dir"]
        if not path.isdir(tmp_dir):
            print(f"Temporary directory does not exist. Creating it: {tmp_dir}")
            os.makedirs(tmp_dir)

        # Parse simulations.
        sims_args = [
            (idx, tot_sims, net, sim, tmp_dir, args["warmup_percent"],
             args["keep_percent"])
            for idx, sim in enumerate(sims)]
        if defaults.SYNC or args["sync"]:
            dat_all = [process_sim(*sim_args) for sim_args in sims_args]
        else:
            with multiprocessing.Pool() as pol:
                # Each element of dat_all corresponds to a single simulation.
                dat_all = pol.starmap(process_sim, sims_args)
        # Throw away results from simulations that could not be parsed.
        dat_all = [dat for dat in dat_all if dat is not None]
        print(f"Discarded {tot_sims - len(dat_all)} simulations!")
        assert dat_all, "No valid simulations found!"
        dat_all, sims = zip(*dat_all)

        dat_all = [utils.load_tmp_file(flp) for flp in dat_all]
    else:
        dat_all, sims = dat

    # Validate data.
    dim_in = None
    dtype_in = None
    dim_out = None
    dtype_out = None
    dim_extra = None
    dtype_extra = None
    scl_grps = None
    for dat_in, dat_out, dat_extra, scl_grps_cur in dat_all:
        dim_in_cur = len(dat_in.dtype.names)
        dim_out_cur = len(dat_out.dtype.names)
        dim_extra_cur = len(dat_extra.dtype.names)
        dtype_in_cur = dat_in.dtype
        dtype_out_cur = dat_out.dtype
        dtype_extra_cur = dat_extra.dtype
        if dim_in is None:
            dim_in = dim_in_cur
        if dim_out is None:
            dim_out = dim_out_cur
        if dim_extra is None:
            dim_extra = dim_extra_cur
        if dtype_in is None:
            dtype_in = dtype_in_cur
        if dtype_out is None:
            dtype_out = dtype_out_cur
        if dtype_extra is None:
            dtype_extra = dtype_extra_cur
        if scl_grps is None:
            scl_grps = scl_grps_cur
        assert dim_in_cur == dim_in, \
            f"Invalid input feature dim: {dim_in_cur} != {dim_in}"
        assert dim_out_cur == dim_out, \
            f"Invalid output feature dim: {dim_out_cur} != {dim_out}"
        assert dim_extra_cur == dim_extra, \
            f"Invalid extra data dim: {dim_extra_cur} != {dim_extra}"
        assert dtype_in_cur == dtype_in, \
            f"Invalud input dtype: {dtype_in_cur} != {dtype_in}"
        assert dtype_out_cur == dtype_out, \
            f"Invalid output dtype: {dtype_out_cur} != {dtype_out}"
        assert dtype_extra_cur == dtype_extra, \
            f"Invalid extra data dtype: {dtype_extra_cur} != {dtype_extra}"
        assert (scl_grps_cur == scl_grps).all(), \
            f"Invalid scaling groups: {scl_grps_cur} != {scl_grps}"
    assert dim_in is not None, "Unable to compute input feature dim!"
    assert dim_out is not None, "Unable to compute output feature dim!"
    assert dim_extra is not None, "Unable to compute extra data dim!"
    assert dtype_in is not None, "Unable to compute input dtype!"
    assert dtype_out is not None, "Unable to compute output dtype!"
    assert dtype_extra is not None, "Unable to compute extra data dtype!"
    assert scl_grps is not None, "Unable to compte scaling groups!"

    # Build combined feature lists.
    dat_in_all, dat_out_all, dat_extra_all, _ = zip(*dat_all)
    # Stack the arrays.
    dat_in_all = np.concatenate(dat_in_all, axis=0)
    dat_out_all = np.concatenate(dat_out_all, axis=0)
    dat_extra_all = np.concatenate(dat_extra_all, axis=0)

    # Convert all instances of -1 (feature value unknown) to the mean
    # for that feature.
    bad_fets = []
    for fet in dat_in_all.dtype.names:
        fet_values = dat_in_all[fet]
        if (fet_values == -1).all():
            bad_fets.append(fet)
            continue
        dat_in_all[fet] = np.where(
            fet_values == -1, np.mean(fet_values), fet_values)
        assert (dat_in_all[fet] != -1).all(), f"Found \"-1\" in feature: {fet}"
    assert not bad_fets, f"Features contain only \"-1\": {bad_fets}"

    # Scale input features. Do this here instead of in process_sim()
    # because all of the features must be scaled using the same
    # parameters.
    dat_in_all, prms_in = scale_fets(dat_in_all, scl_grps, args["standardize"])

    # # Check if any of the data is malformed and discard features if
    # # necessary.
    # fets = []
    # for fet in dat_in_all.dtype.names:
    #     fet_values = dat_in_all[fet]
    #     if ((not np.isnan(fet_values).any()) and
    #             (not np.isinf(fet_values).any())):
    #         fets.append(fet)
    #     else:
    #         print(f"Discarding: {fet}")
    # dat_in_all = dat_in_all[fets]

    return dat_in_all, dat_out_all, dat_extra_all, prms_in


def gen_data(net, args, dat_flp, scl_prms_flp, dat=None, save_data=True):
    """ Generates training data and optionally saves it. """
    dat_in, dat_out, dat_extra, scl_prms = make_datasets(net, args, dat)
    # Save the processed data so that we do not need to process it again.
    if save_data:
        utils.save(dat_flp, dat_in, dat_out, dat_extra)
    # Save scaling parameters.
    print(f"Saving scaling parameters: {scl_prms_flp}")
    with open(scl_prms_flp, "w") as fil:
        json.dump(scl_prms.tolist(), fil)
    return dat_in, dat_out, dat_extra


def split_data(net, dat_in, dat_out, dat_extra, bch_trn, bch_tst,
               use_val=False):
    """
    Divides the input and output data into training, validation, and
    testing sets and constructs data loaders.
    """
    print("Creating train/val/test data...")

    fets = dat_in.dtype.names
    # Keep track of the dtype of dat_extra so that we can recreate it
    # as a structured array.
    extra_dtype = dat_extra.dtype.descr
    # Destroy columns names to make merging the matrices easier. I.e.,
    # convert from structured to regular numpy arrays.
    dat_in = utils.clean(dat_in)
    dat_out = utils.clean(dat_out)
    dat_extra = utils.clean(dat_extra)
    # Shuffle the data to ensure that the training, validation, and
    # test sets are uniformly sampled. To shuffle all of the arrays
    # together, we must first merge them into a combined matrix.
    num_cols_in = dat_in.shape[1]
    merged = np.concatenate(
        (dat_in, dat_out, dat_extra), axis=1)
    np.random.shuffle(merged)
    dat_in = merged[:, :num_cols_in]
    dat_out = merged[:, num_cols_in]
    # Rebuilding dat_extra is more complicated because we need it to
    # be a structed array (for ease of use).
    num_exps = dat_in.shape[0]
    dat_extra = np.empty((num_exps,), dtype=extra_dtype)
    num_cols = merged.shape[1]
    num_cols_extra = num_cols - (num_cols_in + 1)
    extra_names = dat_extra.dtype.names
    num_cols_extra_expected = len(extra_names)
    assert num_cols_extra == num_cols_extra_expected, \
        (f"Error while reassembling \"dat_extra\". {num_cols_extra} columns "
         f"does not match {num_cols_extra_expected} expected columns: "
         f"{extra_names}")
    for name, merged_idx in zip(extra_names, range(num_cols_in + 1, num_cols)):
        dat_extra[name] = merged[:, merged_idx]

    # 50% for training, 20% for validation, 30% for testing.
    num_val = int(round(num_exps * 0.2)) if use_val else 0
    num_tst = int(round(num_exps * 0.3))
    print((f"    Data - train: {num_exps - num_val - num_tst}, val: {num_val}, "
           f"test: {num_tst}"))
    # Validation.
    dat_val_in = dat_in[:num_val]
    dat_val_out = dat_out[:num_val]
    dat_val_extra = dat_extra[:num_val]
    # Testing.
    dat_tst_in = dat_in[num_val:num_val + num_tst]
    dat_tst_out = dat_out[num_val:num_val + num_tst]
    dat_tst_extra = dat_extra[num_val:num_val + num_tst]
    # Training.
    dat_trn_in = dat_in[num_val + num_tst:]
    dat_trn_out = dat_out[num_val + num_tst:]
    dat_trn_extra = dat_extra[num_val + num_tst:]

    # Create the dataloaders.
    dataset_trn = utils.Dataset(fets, dat_trn_in, dat_trn_out, dat_trn_extra)
    ldr_trn = (
        torch.utils.data.DataLoader(
            dataset_trn, batch_size=bch_tst, shuffle=True, drop_last=False)
        if isinstance(net, models.SvmSklearnWrapper)
        else torch.utils.data.DataLoader(
            dataset_trn,
            batch_sampler=utils.BalancedSampler(
                dataset_trn, bch_trn, drop_last=False)))
    ldr_val = (
        torch.utils.data.DataLoader(
            utils.Dataset(fets, dat_val_in, dat_val_out, dat_val_extra),
            batch_size=bch_tst, shuffle=False, drop_last=False)
        if use_val else None)
    ldr_tst = torch.utils.data.DataLoader(
        utils.Dataset(fets, dat_tst_in, dat_tst_out, dat_tst_extra),
        batch_size=bch_tst, shuffle=False, drop_last=False)
    return ldr_trn, ldr_val, ldr_tst


def init_hidden(net, bch, dev):
    """
    Initialize the hidden state. The hidden state is what gets built
    up over time as the LSTM processes a sequence. It is specific to a
    sequence, and is different than the network's weights. It needs to
    be reset for every new sequence.
    """
    hidden = net.init_hidden(bch)
    hidden[0].to(dev)
    hidden[1].to(dev)
    return hidden


def inference(ins, labs, net_raw, dev,
              hidden=(torch.zeros(()), torch.zeros(())), los_fnc=None):
    """
    Runs a single inference pass. Returns the output of net, or the
    loss if los_fnc is not None.
    """
    # Move input and output data to the proper device.
    ins = ins.to(dev)
    labs = labs.to(dev)

    if isinstance(net_raw, models.Lstm):
        # LSTMs want the sequence length to be first and the batch
        # size to be second, so we need to flip the first and
        # second dimensions:
        #   (batch size, sequence length, LSTM.in_dim) to
        #   (sequence length, batch size, LSTM.in_dim)
        ins = ins.transpose(0, 1)
        # Reduce the labels to a 1D tensor.
        # TODO: Explain this better.
        labs = labs.transpose(0, 1).view(-1)
        # The forward pass.
        out, hidden = net_raw(ins, hidden)
    else:
        # The forward pass.
        out = net_raw(ins)
    if los_fnc is None:
        return out, hidden
    return los_fnc(out, labs), hidden


def train(net, num_epochs, ldr_trn, ldr_val, dev, ely_stp, val_pat_max, out_flp,
          val_imp_thresh, tim_out_s, opt_params):
    """ Trains a model. """
    print("Training...")
    los_fnc = net.los_fnc()
    opt = net.opt(net.net.parameters(), **opt_params)
    # If using early stopping, then this is the lowest validation loss
    # encountered so far.
    los_val_min = None
    # If using early stopping, then this tracks the *remaining* validation
    # patience (initially set to the maximum amount of patience). This is
    # decremented for every validation pass that does not improve the
    # validation loss by at least val_imp_thresh percent. When this reaches
    # zero, training aborts.
    val_pat = val_pat_max
    # The number of batches per epoch.
    num_bchs_trn = len(ldr_trn)
    # Print a lot statement every few batches.
    if LOGS_PER_EPC == 0:
        # Disable logging.
        bchs_per_log = sys.maxsize
    else:
        bchs_per_log = math.ceil(num_bchs_trn / LOGS_PER_EPC)
    # Perform a validation pass every few batches.
    assert not ely_stp or VALS_PER_EPC > 0, \
        f"Early stopping configured with erroneous VALS_PER_EPC: {VALS_PER_EPC}"
    bchs_per_val = math.ceil(num_bchs_trn / VALS_PER_EPC)
    if ely_stp:
        print(f"Will validate after every {bchs_per_val} batches.")

    tim_srt_s = time.time()
    # Loop over the dataset multiple times...
    for epoch_idx in range(num_epochs):
        tim_del_s = time.time() - tim_srt_s
        if tim_out_s != 0 and tim_del_s > tim_out_s:
            print((f"Training timed out after after {epoch_idx} epochs "
                   f"({tim_del_s:.2f} seconds)."))
            break

        # For each batch...
        for bch_idx_trn, (ins, labs) in enumerate(ldr_trn, 0):
            if bch_idx_trn % bchs_per_log == 0:
                print(f"Epoch: {epoch_idx + 1:{f'0{len(str(num_epochs))}'}}/"
                      f"{'?' if ely_stp else num_epochs}, batch: "
                      f"{bch_idx_trn + 1:{f'0{len(str(num_bchs_trn))}'}}/"
                      f"{num_bchs_trn}", end=" ")
            # Initialize the hidden state for every new sequence.
            hidden = init_hidden(net, bch=ins.size()[0], dev=dev)
            # Zero out the parameter gradients.
            opt.zero_grad()
            loss, hidden = inference(ins, labs, net.net, dev, hidden, los_fnc)
            # The backward pass.
            loss.backward()
            opt.step()
            if bch_idx_trn % bchs_per_log == 0:
                print(f"    Training loss: {loss:.5f}")

            # Run on validation set, print statistics, and (maybe) checkpoint
            # every VAL_PER batches.
            if ely_stp and not bch_idx_trn % bchs_per_val:
                print("    Validation pass:")
                # For efficiency, convert the model to evaluation mode.
                net.net.eval()
                with torch.no_grad():
                    los_val = 0
                    for bch_idx_val, (ins_val, labs_val) in enumerate(ldr_val):
                        print(
                            "    Validation batch: "
                            f"{bch_idx_val + 1}/{len(ldr_val)}")
                        # Initialize the hidden state for every new sequence.
                        hidden = init_hidden(net, bch=ins.size()[0], dev=dev)
                        los_val += inference(
                            ins_val, labs_val, net.net, dev, hidden,
                            los_fnc)[0].item()
                # Convert the model back to training mode.
                net.net.train()

                if los_val_min is None:
                    los_val_min = los_val
                # Calculate the percent improvement in the validation loss.
                prc = (los_val_min - los_val) / los_val_min * 100
                print(f"    Validation error improvement: {prc:.2f}%")

                # If the percent improvement in the validation loss is greater
                # than a small threshold, then take this as the new best version
                # of the model.
                if prc > val_imp_thresh:
                    # This is the new best version of the model.
                    los_val_min = los_val
                    # Reset the validation patience.
                    val_pat = val_pat_max
                    # Save the new best version of the model. Convert the
                    # model to Torch Script first.
                    torch.jit.save(torch.jit.script(net.net), out_flp)
                else:
                    val_pat -= 1
                    if path.exists(out_flp):
                        # Resume training from the best model.
                        net.net = torch.jit.load(out_flp)
                        net.net.to(dev)
                if val_pat <= 0:
                    print(f"Stopped after {epoch_idx + 1} epochs")
                    return net
    if not ely_stp:
        # Save the final version of the model. Convert the model to Torch Script
        # first.
        print(f"Saving final model: {out_flp}")
        torch.jit.save(torch.jit.script(net.net), out_flp)
    return net


def test(net, ldr_tst, dev):
    """ Tests a model. """
    print("Testing...")
    # The number of testing samples that were predicted correctly.
    num_correct = 0
    # Total testing samples.
    total = 0
    num_bchs_tst = len(ldr_tst)
    # For efficiency, convert the model to evaluation mode.
    net.net.eval()
    with torch.no_grad():
        for bch_idx, (ins, labs) in enumerate(ldr_tst):
            print(f"Test batch: {bch_idx + 1:{f'0{len(str(num_bchs_tst))}'}}/"
                  f"{num_bchs_tst}")
            if isinstance(net, models.LstmWrapper):
                bch_tst, seq_len, _ = ins.size()
            else:
                bch_tst, _ = ins.size()
                seq_len = 1
            # Initialize the hidden state for every new sequence.
            hidden = init_hidden(net, bch=bch_tst, dev=dev)
            # Run inference. The first element of the output is the
            # number of correct predictions.
            num_correct += inference(
                ins, labs, net.net, dev, hidden, los_fnc=net.check_output)[0]
            total += bch_tst * seq_len
    # Convert the model back to training mode.
    net.net.train()
    acc_tst = num_correct / total
    print(f"Test accuracy: {acc_tst * 100:.2f}%")
    return acc_tst


def run_sklearn(args, dat_in, dat_out, dat_extra, out_dir, out_flp):
    """
    Trains an sklearn model according to the supplied parameters. Returns the
    test error (lower is better).
    """
    # Construct the model.
    print("Building model...")
    net = models.MODELS[args["model"]]()
    net.new(**{param: args[param] for param in net.params})
    # Split the data into training, validation, and test loaders.
    ldr_trn, _, ldr_tst = split_data(
        net, dat_in, dat_out, dat_extra, args["train_batch"],
        args["test_batch"])
    # Training.
    print("Training...")
    tim_srt_s = time.time()
    net.train(*(ldr_trn.dataset.raw()[1:3]))
    tim_trn_s = time.time() - tim_srt_s
    print(f"Finished training - time: {tim_trn_s:.2f} seconds")
    del ldr_trn
    # Save the model.
    print(f"Saving final model: {out_flp}")
    with open(out_flp, "wb") as fil:
        pickle.dump(net.net, fil)
    # Testing.
    print("Testing...")
    tim_srt_s = time.time()
    # Select the first return value, which is the overall accuracy.
    acc_tst = net.test(
        *ldr_tst.dataset.raw(),
        graph_prms={
            "out_dir": out_dir, "sort_by_unfairness": True, "dur_s": None})[0]
    print(f"Finished testing - time: {time.time() - tim_srt_s:.2f} seconds")
    del ldr_tst
    return acc_tst, tim_trn_s


def run_torch(args, dat_in, dat_out, dat_extra, out_dir, out_flp):
    """
    Trains a PyTorch model according to the supplied parameters. Returns the
    test error (lower is better).
    """
    # Instantiate and configure the network. Move it to the proper device.
    net = models.MODELS[args["model"]]()
    net.new()
    num_gpus = torch.cuda.device_count()
    num_gpus_to_use = args["num_gpus"]
    if num_gpus >= num_gpus_to_use > 1:
        net.net = torch.nn.DataParallel(net.net)
    dev = torch.device("cuda:0" if num_gpus >= num_gpus_to_use > 0 else "cpu")
    net.net.to(dev)

    # Split the data into training, validation, and test loaders.
    ldr_trn, ldr_val, ldr_tst = split_data(
        net, dat_in, dat_out, dat_extra, args["train_batch"],
        args["test_batch"])

    # Explicitly move the training (and maybe validation) data to the target
    # device.
    ldr_trn.dataset.to(dev)
    ely_stp = args["early_stop"]
    if ely_stp:
        ldr_val.dataset.to(dev)

    # Training.
    tim_srt_s = time.time()
    net = train(
        net, args["epochs"], ldr_trn, ldr_val, dev, args["early_stop"],
        args["val_patience"], out_flp, args["val_improvement_thresh"],
        args["timeout_s"],
        opt_params={param: args[param] for param in net.params})
    tim_trn_s = time.time() - tim_srt_s
    print(f"Finished training - time: {tim_trn_s:.2f} seconds")

    # Explicitly delete the training and validation data so that they are
    # removed from the target device.
    del ldr_trn
    del ldr_val
    # This is necessary for the GPU memory to be released.
    torch.cuda.empty_cache()

    # Read the best version of the model from disk.
    net.net = torch.jit.load(out_flp)
    net.net.to(dev)

    # Testing.
    ldr_tst.dataset.to(dev)
    tim_srt_s = time.time()
    acc_tst = test(net, ldr_tst, dev)
    print(f"Finished testing - time: {time.time() - tim_srt_s:.2f} seconds")
    del ldr_tst
    torch.cuda.empty_cache()
    return acc_tst, tim_trn_s


def prepare_args(args_):
    """ Updates the default arguments with the specified values. """
    # Initially, accept all default values. Then, override the defaults with
    # any manually-specified values. This allows the caller to specify values
    # only for parameters that they care about while ensuring that all
    # parameters have values.
    args = copy.copy(defaults.DEFAULTS)
    args.update(args_)
    return args


def run_trials(args):
    """
    Run args["conf_trials"] trials and survive args["max_attempts"] failed
    attempts.
    """
    print(f"Arguments: {args}")

    if args["no_rand"]:
        utils.set_rand_seed()

    out_dir = args["out_dir"]
    if not path.isdir(out_dir):
        print(f"Output directory does not exist. Creating it: {out_dir}")
        os.makedirs(out_dir)
    net_tmp = models.MODELS[args["model"]]()
    # Verify that the necessary supplemental parameters are present.
    for param in net_tmp.params:
        assert param in args, f"\"{param}\" not in args: {args}"
    # Assemble the output filepath.
    out_flp = path.join(
        args["out_dir"],
        (utils.args_to_str(args, order=sorted(defaults.DEFAULTS.keys()))
        ) + (
            # Determine the proper extension based on the type of
            # model.
            ".pickle" if isinstance(net_tmp, models.SvmSklearnWrapper)
            else ".pth"))
    # If custom features are specified, then overwrite the model's
    # default features.
    fets = args["features"]
    if fets:
        net_tmp.in_spc = fets
    else:
        args["features"] = net_tmp.in_spc
    # If a trained model file already exists, then delete it.
    if path.exists(out_flp):
        os.remove(out_flp)

    # Load or geenrate training data.
    dat_flp = path.join(out_dir, "data.npz")
    scl_prms_flp = path.join(out_dir, "scale_params.json")
    # Check for the presence of both the data and the scaling
    # parameters because the resulting model is useless without the
    # proper scaling parameters.
    if (not args["regen_data"] and path.exists(dat_flp) and
            path.exists(scl_prms_flp)):
        print("Found existing data!")
        dat_in, dat_out, dat_extra = utils.load(dat_flp)
        dat_in_shape = dat_in.shape
        dat_out_shape = dat_out.shape
        assert dat_in_shape[0] == dat_out_shape[0], \
            f"Data has invalid shapes! in: {dat_in_shape}, out: {dat_out_shape}"
    else:
        print("Regenerating data...")
        dat_in, dat_out, dat_extra = gen_data(
            net_tmp, args, dat_flp, scl_prms_flp)
    print(f"Number of input features: {len(dat_in.dtype.names)}")

    # Visualaize the ground truth data.
    utils.visualize_classes(net_tmp, dat_out)

    # TODO: Parallelize attempts.
    trls = args["conf_trials"]
    apts = 0
    apts_max = args["max_attempts"]
    ress = []
    while trls > 0 and apts < apts_max:
        apts += 1
        res = (
            run_sklearn
            if isinstance(net_tmp, models.SvmSklearnWrapper)
            else run_torch)(args, dat_in, dat_out, dat_extra, out_dir, out_flp)
        if res[0] == 100:
            print(
                (f"Training failed (attempt {apts}/{apts_max}). Trying again!"))
        else:
            ress.append(res)
            trls -= 1
    if ress:
        print(("Resulting accuracies: "
               f"{', '.join([f'{acc:.2f}' for acc, _ in ress])}"))
        max_acc, tim_s = max(ress, key=lambda p: p[0])
        print(f"Maximum accuracy: {max_acc:.2f}")
        # Return the minimum error instead of the maximum accuracy.
        return 1 - max_acc, tim_s
    print(f"Model cannot be trained with args: {args}")
    return float("NaN"), float("NaN")


def run_cnf(cnf, gate_func=None, post_func=None):
    """ Evaluate a single configuration. """
    func = run_trials
    # Optionally decide whether to run a configuration.
    if gate_func is not None:
        func = functools.partial(gate_func, func=func)
    res = func(cnf)
    # Optionally process the output of each configuration.
    if post_func is not None:
        res = post_func(cnf, res)
    return res


def run_cnfs(cnfs, sync=False, gate_func=None, post_func=None):
    """
    Evaluates many configurations. Assumes that the arguments have already been
    processed with prepare_args().
    """
    num_cnfs = len(cnfs)
    print(f"Training {num_cnfs} configurations.")
    # The configurations themselves should execute synchronously if
    # and only if sync is False or the configuration is explicity
    # configured to run synchronously.
    cnfs = zip(
        [{**cnf,
          "sync": (not sync) or cnf.get("sync", defaults.DEFAULTS["sync"])}
         for cnf in cnfs],
        [gate_func,] * num_cnfs, [post_func,] * num_cnfs)

    if defaults.SYNC:
        res = [run_cnf(*cnf) for cnf in cnfs]
    else:
        with multiprocessing.Pool(processes=3) as pol:
            res = pol.starmap(run_cnf, cnfs)
    return res


def main():
    """ This program's entrypoint. """
    # Parse command line arguments.
    psr = argparse.ArgumentParser(description="An LSTM training framework.")
    psr.add_argument(
        "--graph", action="store_true",
        help=("If the model is an sklearn model, then analyze and graph the "
              "testing results."))
    psr, psr_verify = cl_args.add_training(psr)
    args = vars(psr_verify(psr.parse_args()))
    # Verify that all arguments are reflected in defaults.DEFAULTS.
    for arg in args.keys():
        assert arg in defaults.DEFAULTS, \
            f"Argument {arg} missing from defaults.DEFAULTS!"
    run_trials(prepare_args(args))


if __name__ == "__main__":
    main()
