""" Utility functions. """

import collections
import json
import math
import os
from os import path
import pickle
import random
import struct
import sys
import time
import zipfile

from matplotlib import pyplot as plt
import numpy as np
import scapy
import scapy.layers.l2
import scapy.layers.inet
import scapy.utils
from scipy import stats
from scipy import cluster
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import inspection
import torch

import defaults
import features


# Values considered unsafe for division and min().
UNSAFE = {-1, 0}


class Dataset(torch.utils.data.Dataset):
    """ A simple Dataset that wraps arrays of input and output features. """

    def __init__(self, fets, dat_in, dat_out, dat_extra):
        """
        fets: List of input feature names, corresponding to the columns of
            dat_in.
        dat_in: Numpy array of input data.
        dat_out: Numpy array of output data. Assumed to have a single practical
            dimension only (e.g., dat_out should be of shape (X,), or (X, 1)).
        dat_extra: Numpy array of extra data.
        """
        super(Dataset).__init__()
        shp_in = dat_in.shape
        shp_out = dat_out.shape
        assert shp_in[0] == shp_out[0], \
            f"Mismatched dat_in ({shp_in}) and dat_out ({shp_out})!"
        num_fets = len(fets)
        assert shp_in[1] == num_fets, \
            f"Mismatched dat_in ({shp_in}) and fets (len: {num_fets})"

        self.fets = fets
        # Convert the numpy arrays to Torch tensors.
        self.dat_in = torch.tensor(dat_in, dtype=torch.float)
        # Reshape the output into a 1D array first, because
        # CrossEntropyLoss expects a single value. The dtype must be
        # long because the loss functions expect longs.
        self.dat_out = torch.tensor(
            dat_out.reshape(shp_out[0]), dtype=torch.long)
        # Do not convert dat_extra to a Torch tensor because it will
        # not interact with models.
        self.dat_extra = dat_extra

    def to(self, dev):
        """ Move the entire dataset to the target device. """
        try:
            # This will fail if there is insufficient memory.
            self.dat_in = self.dat_in.to(dev)
            self.dat_out = self.dat_out.to(dev)
        except RuntimeError:
            print(f"Warning:: Unable to move dataset to device: {dev}")
            # In case the input data was moved successfully but there
            # was insufficient device memory for the output data, move
            # the input data back to main memory.
            self.dat_in = self.dat_in.to(torch.device("cpu"))

    def __len__(self):
        """ Returns the number of items in this Dataset. """
        return self.dat_in.size()[0]

    def __getitem__(self, idx):
        """ Returns a specific (input, output) pair from this Dataset. """
        assert torch.utils.data.get_worker_info() is None, \
            "This Dataset does not support being loaded by multiple workers!"
        return self.dat_in[idx], self.dat_out[idx]

    def raw(self):
        """ Returns the raw data underlying this dataset. """
        return self.fets, self.dat_in, self.dat_out, self.dat_extra


class BalancedSampler:
    """
    A batching sampler that creates balanced batches. The batch size
    must be evenly divided by the number of classes. This does not
    inherit from any of the existing Torch Samplers because it does
    not require any of their functionalty. Instead, this is a wrapper
    for many other Samplers, one for each class.
    """

    def __init__(self, dataset, batch_size, drop_last, drop_popular):
        assert isinstance(dataset, Dataset), \
            "Dataset must be an instance of utils.Dataset."
        _, _, dat_out, _ = dataset.raw()
        assert_tensor(dat_out=dat_out)

        # Determine the unique classes.
        clss = set(np.unique(dat_out))
        num_clss = len(clss)

        if batch_size is None:
            # If we do not specify a batch size, then prune the dataset so that
            # the length of the dataset is a multiple of the number of classes.
            num_samples = dat_out.size()[0]
            to_drop = num_samples % num_clss
            batch_size = num_samples - to_drop
            dat_out = dat_out[:batch_size]
            # Recalculate the classes, in case one of the classes was only
            # represented in the tail of dat_out that we just removed. If that
            # edge case occurs, then the assert statements below will still
            # guarantee safety.
            clss = set(np.unique(dat_out))
            num_clss = len(clss)
            print(
                f"Dropped {to_drop} samples to enable BalancedSampler with no "
                "batch size.")

        assert batch_size >= num_clss, \
            (f"The batch size ({batch_size}) must be at least as large as the "
             f"number of classes ({num_clss})!")
        # The batch size must be evenly divisible by the number of classes.
        assert batch_size % num_clss == 0, \
            (f"The number of classes ({num_clss}) must evenly divide the batch "
             f"size ({batch_size})!")

        print("Balancing classes...")
        # Find the indices for each class.
        clss_idxs = {cls: torch.where(dat_out == cls)[0] for cls in clss}

        if drop_popular:
            # Determine the number of examples in the least populous class.
            target_examples = min(
                cls_idxs.size()[0] for cls_idxs in clss_idxs.values())
            # Remove samples from the popular classes.
            for cls, cls_idxs in clss_idxs.items():
                num_examples = cls_idxs.size()[0]
                # If this class has too many examples...
                if num_examples > target_examples:
                    # Select a subset of the samples.
                    clss_idxs[cls] = cls_idxs[torch.multinomial(
                        # Sample from the existing examples using a uniform
                        # distribution.
                        torch.ones((num_examples,)),
                        num_samples=target_examples,
                        # Do not sample with replacement because num_samples is
                        # guaranteed to be greater than or equal to
                        # target_samples.
                        replacement=False)]
                    print(
                        f"\tRemoved {num_examples - target_examples} examples "
                        f"from class {cls}.")

        else:
            # Determine the number of examples in the most populous class.
            target_examples = max(
                cls_idxs.size()[0] for cls_idxs in clss_idxs.values())
            # Generate new samples to fill in under-represented classes.
            for cls, cls_idxs in clss_idxs.items():
                num_examples = cls_idxs.size()[0]
                # If this class has insufficient examples...
                if num_examples < target_examples:
                    new_examples = target_examples - num_examples
                    # Duplicate existing examples to make this class balanced.
                    # Append the duplicated examples to the true examples.
                    clss_idxs[cls] = torch.cat(
                        (cls_idxs,
                         cls_idxs[torch.multinomial(
                             # Sample from the existing examples using a uniform
                             # distribution.
                             torch.ones((num_examples,)),
                             num_samples=new_examples,
                             # Sample with replacement in case the number of new
                             # examples is greater than the number of existing
                             # examples.
                             replacement=True)]),
                        dim=0)
                    print(f"\tAdded {new_examples} examples to class {cls}.")

        # Create a BatchSampler iterator for each class.
        examples_per_cls = batch_size // num_clss
        self.samplers = {
            cls: torch.utils.data.BatchSampler(
                torch.utils.data.SubsetRandomSampler(cls_idxs),
                examples_per_cls, drop_last)
            for cls, cls_idxs in clss_idxs.items()}
        # After __iter__() is called, this will contain an iterator for each
        # class.
        self.iters = {}
        self.num_batches = target_examples // examples_per_cls

    def __iter__(self):
        # Create an iterator for each class.
        self.iters = {
            cls: iter(sampler) for cls, sampler in self.samplers.items()}
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        # Pull examples from each class and merge them into a single list.
        idxs = [idx for it in self.iters.values() for idx in next(it)]
        random.shuffle(idxs)
        return idxs


class Exp():
    """ Describes the parameters of a simulation. """

    def __init__(self, sim):
        if "/" in sim:
            sim = path.basename(sim)
        self.name = sim
        toks = sim.split("-")
        if sim.endswith(".tar.gz"):
            # unfair-pcc-cubic-8bw-30rtt-64q-1pcc-1cubic-100s-20201118T114242.tar.gz
            # Remove ".tar.gz" from the last token.
            toks[-1] = toks[-1][:-7]
            # Update sim.name.
            self.name = self.name[:-7]
        elif sim.endswith(".npz"):
            # Remove ".npz" from the last token.
            toks[-1] = toks[-1][:-4]
            # Update sim.name.
            self.name = self.name[:-4]
        # unfair-pcc-cubic-8bw-30rtt-64q-1pcc-1cubic-100s-20201118T114242
        (_, self.cca_1_name, self.cca_2_name, bw_Mbps, rtt_ms, queue_p,
         cca_1_flws, cca_2_flws, end_time, _) = toks
        # Link bandwidth (Mbps).
        self.bw_Mbps = float(bw_Mbps[:-2])
        self.bw_bps = self.bw_Mbps * 1e6
        # Bottleneck router delay (us).
        self.rtt_us = float(rtt_ms[:-3]) * 1000
        # Bandwidth-delay product (bits).
        self.bdp_b = self.bw_Mbps * self.rtt_us
        # Queue size (packets).
        self.queue_p = float(queue_p[:-1])
        # Queue size (multiples of the BDP).
        self.queue_bdp = self.queue_p / (self.bdp_b / 8 / 1514)
        # Number of CCA 1 flows.
        self.cca_1_flws = int(cca_1_flws[:-(len(self.cca_1_name))])
        # Number of CCA 2 flows.
        self.cca_2_flws = int(cca_2_flws[:-(len(self.cca_2_name))])
        # The total number of flows.
        self.tot_flws = self.cca_1_flws + self.cca_2_flws
        # Experiment duration (s).
        self.dur_s = int(end_time[:-1])
        # Largest RTT that this experiment should experiment, based on the size
        # of the bottleneck queue and the RTT.
        self.calculated_max_rtt_us = (self.queue_bdp + 1) * self.rtt_us
        # Fair share bandwidth for each flow.
        self.target_per_flow_bw_Mbps = (
            self.bw_Mbps / (self.cca_1_flws + self.cca_2_flws))


def args_to_str(args, order, which):
    """
    Converts the provided arguments dictionary to a string, using the
    keys in order to determine the order of arguments in the
    string.
    """
    assert which in {"model", "data"}
    for key in order:
        assert key in args, f"Key {key} not in args: {args}"
    return "-".join(
        [str(args[key]) for key in order
         if key not in (
            defaults.ARGS_TO_IGNORE_MODEL if which == "model" else
            defaults.ARGS_TO_IGNORE_DATA)])


def str_to_args(args_str, order, which):
    """
    Converts the provided string of arguments to a dictionary, using
    the keys in order to determine the identity of each argument in the
    string.
    """
    assert which in {"model", "data"}
    # Remove extension and split on "-".
    toks = ".".join(args_str.split(".")[:-1]).split("-")
    # Remove elements of order that args_to_str() does not use when
    # encoding strings.
    order = [
        key for key in order if key not in (
            defaults.ARGS_TO_IGNORE_MODEL if which == "model" else
            defaults.ARGS_TO_IGNORE_DATA)]
    num_toks = len(toks)
    num_order = len(order)
    assert num_toks == num_order, \
        (f"Mismatched tokens ({num_toks}) and order ({num_order})! "
         "tokens: {toks}, order: {order}")
    parsed = {}
    for arg, tok in zip(order, toks):
        try:
            parsed_val = float(tok)
        except ValueError:
            parsed_val = tok
        parsed[arg] = parsed_val
    return parsed


def parse_packets(flp, flw_to_cca):
    """
    Parses a PCAP file. Considers packets between a specified client and server
    using specified ports only.

    Returns a dictionary mapping flow to a tuple containing two lists, one for
    data packets and one for ACK packets:
        {
              (client port, server port) :
                  ([ list of data packets ], [ list of ACK packets ])
        }

    Each packet is a tuple of the form:
         (sequence number, timestamp (us),
          TCP timestamp option TSval, TCP timestamp option TSecr,
          TCP payload size (B), total packet size (B))
    """
    print(f"\tParsing PCAP: {flp}")
    # Use list() to read the pcap file all at once (minimize seeks).
    pkts = list(enumerate(scapy.utils.RawPcapReader(flp)))
    num_pkts = len(pkts)

    def make_empty():
        """ Make an empty numpy array to store the packets. """
        return np.full((num_pkts,), -1, dtype=features.PARSE_PACKETS_FETS)

    def remove_unused_rows(arr):
        """
        Returns a filtered array with unused rows removed. A row is unused if
        all of its columns are -1. As an optimization, we check the second
        column (timestamp) in each row only because the timestamp should never
        be unknown because it comes from PCAP.
        """
        return arr[arr[features.ARRIVAL_TIME_FET] != -1]

    # Format described above. In this form, the arrays will be sparse. Unused
    # rows will be removed later.
    flw_to_pkts = {
        flw_ports: (make_empty(), make_empty())
        for flw_ports in flw_to_cca.keys()}
    for idx, (pkt_dat, pkt_mdat) in pkts:
        ether = scapy.layers.l2.Ether(pkt_dat)
        # Assume that this is a TCP/IP packet.
        ip = ether[scapy.layers.inet.IP]
        is_tcp = scapy.layers.inet.TCP in ether
        trans = ether[
            scapy.layers.inet.TCP if is_tcp else scapy.layers.inet.UDP]
        # Determine this packet's direction. Assume that the client IP address
        # if 192.0.0.4 and the server IP address is 192.0.0.2. Assume that all
        # packets are between the client and server.
        if ip.src[-1] == "4":
            dir_idx = 0
            flw = (trans.sport, trans.dport)
        else:
            dir_idx = 1
            flw = (trans.dport, trans.sport)
        # Assume that the packets are between the relevent machines. Only check
        # the ports.
        if flw in flw_to_pkts:
            # Decode the sequence number and timestamp info.
            seq = -1
            ts = (-1, -1)
            if is_tcp:
                seq = trans.seq
                trans_header_len = trans.dataofs << 2
                # Decode the TCP Timestamp option.
                if (len(trans.options) >= 3 and
                        trans.options[2][0] == "Timestamp"):
                    # Fast path: it is usually the third option.
                    ts = trans.options[2][1]
                else:
                    # Slow path: check all of the options.
                    for option_name, option in trans.options:
                        if option_name == "Timestamp":
                            ts = option
                            break
            else:
                # Start with the UDP header size.
                trans_header_len = 8
                cca = flw_to_cca[flw]
                if cca == "copa":
                    # Add the Copa header size to the UDP header size.
                    trans_header_len += defaults.COPA_HEADER_SIZE_B
                    # The Copa header is the first part of the UDP payload.
                    #     int seq_num;
	                #     int flow_id;
	                #     int src_id;
	                #     double sender_timestamp;  // milliseconds
	                #     double receiver_timestamp;  // milliseconds
                    seq, _, _, sender_ts, receiver_ts = struct.unpack(
                        defaults.COPA_HEADER_FMT,
                        # Convert the transport payload to bytes and then select
                        # the Copa header only.
                        bytes(trans.payload)[:defaults.COPA_HEADER_SIZE_B])
                    if seq == -1:
                        # This is a connection-establishment packet. Skip it.
                        continue
                    # Convert from milliseconds to microsecods and then convert
                    # from a double to an int.
                    ts = (
                        int(round(sender_ts * 1000)),
                        int(round(receiver_ts * 1000)))

                    # Furthermore, the Copa packet data includes:
                    #     Time sent_time;
                    #     Time intersend_time;
                    #     Time intersend_time_vel;
                    #     Time rtt;
                    #     double prev_avg_sending_rate;
                    # These may be of use.
                elif cca == "vivace":
                    # PCC Vivace is based on UDT: UDP-based Data Transfer
                    # Protocol.
                    #
                    # See https://tools.ietf.org/pdf/draft-gg-udt-03.pdf
                    trans_header_len += 16
                    payload = bytes(trans.payload)
                    first = int.from_bytes(payload[:4], byteorder="big")
                    if (first & 0x80000000) >> 31:
                        # Type code of 1 = UDT control packet.
                        if dir_idx == 0:
                            # Client -> server control packet. Skip it.
                            continue
                        if (first & 0x7fff0000) >> 16 == 2:
                            # ACK.
                            trans_header_len += 24
                            seq = int.from_bytes(
                                payload[16:20], byteorder="big")
                            ts = (
                                # UDT ACKs contain the RTT, so extract that as
                                # the first ts value.
                                int.from_bytes(
                                    payload[20:24], byteorder="big"),
                                # The second ts field is unused.
                                -1)
                        else:
                            # One of the other seven types of control
                            # packets. Skip it.
                            continue
                    else:
                        # Type code of 0 = UDT data packet.
                        seq = first & 0x7fffffff

            flw_to_pkts[flw][dir_idx][idx] = (
                # Sequence number.
                seq,
                # Timestamp. Not using parse_time_us for efficiency purpose. Use
                # 1000000 instead of 1e6 to avoid converting floats.
                pkt_mdat.sec * 1000000 + pkt_mdat.usec,
                # Timestamp option.
                ts[0],
                ts[1],
                # Transport payload. Length of the IP packet minus the length of
                # the IP header minus the length of the transport header.
                ip.len - (ip.ihl << 2) - trans_header_len,
                # Total packet size.
                pkt_mdat.wirelen)

    # Remove unused rows.
    for flw in flw_to_pkts.keys():
        data, ack = flw_to_pkts[flw]
        flw_to_pkts[flw] = (remove_unused_rows(data), remove_unused_rows(ack))

    # Verify packet count.
    tot_pkts = sum(sum(
        ((dat_pkts.shape[0], ack_pkts.shape[0])
         for dat_pkts, ack_pkts in flw_to_pkts.values()),
        ()))
    assert tot_pkts <= num_pkts, \
        f"Found more packets than exist ({tot_pkts} > {num_pkts}): {flp}"
    discarded_pkts = num_pkts - tot_pkts
    print(
        f"\tDiscarded packets: {discarded_pkts} "
        f"({discarded_pkts / num_pkts * 100:.2f}%)")

    return flw_to_pkts


def parse_q_stats(line):
    """
    Parses a "stats" line of a BESS queue log. Line should be of the form:
        ( "stats", src port, enqueued, dequeued, dropped )
    """
    return (
        ("stats",) +
        tuple(
            int(tok, 16) if tok.startswith("0x") else int(tok)
            for tok in line.split(":")[1].split(",")))


def parse_q_enq_deq(line):
    """
    Parses a packet log line of a BESS queue log. Line should be of the form:
        ( "enq" or "deq", time ns, src port, seq, payload B, qsize, dropped,
          queued, batch size )
    """
    (event, time_ns, src_port, seq, payload_B, qsize, dropped, queued,
     batch_size) = [
         int(tok, 16) if tok.startswith("0x") else int(tok)
         for tok in line.split(",")]

    event_options = {0, 1}
    assert event in event_options, f"Event \"{event}\" not in {event_options}"
    if event == 0:
        event = "enq"
    else:
        event = "deq"

    return (
        event, time_ns / 1e3, src_port, seq, payload_B, qsize, dropped, queued,
        batch_size)


def parse_queue_log(flp):
    """
    Parses the BESS queue log. Returns a list of tuples. See parse_q_stats() and
    parse_q_enq_deq() for details on the tuple format.
    """
    print(f"\tParsing queue log: {flp}")
    with open(flp, "r") as fil:
        q_log = list(fil)
    return [
        parse_q_stats(line) if line.startswith("stats")
        else parse_q_enq_deq(line)
        for line in q_log if line.strip() != ""]


def scale(val, min_in, max_in, min_out, max_out):
    """
    Scales val, which is from the range [min_in, max_in], to the range
    [min_out, max_out].
    """
    assert min_in != max_in, "Divide by zero!"
    return min_out + (val - min_in) * (max_out - min_out) / (max_in - min_in)


def scale_all(dat, scl_prms, min_out, max_out, standardize):
    """
    Uses the provided scaling parameters to scale the columns of
    dat. If standardize is False, then the values are rescaled to the
    range [min_out, max_out].
    """
    dat_dtype = dat.dtype
    fets = dat_dtype.names
    num_scl_prms = len(scl_prms)
    assert len(fets) == num_scl_prms, \
        (f"Mismatching dtype ({fets}) and number of scale parameters "
         f"({num_scl_prms})!")
    new = np.empty(dat.shape, dtype=dat_dtype)
    for idx, fet in enumerate(fets):
        prm_1, prm_2 = scl_prms[idx]
        new[fet] = (
            (dat[fet] - prm_1) / prm_2 if standardize else
            scale(
                dat[fet], min_in=prm_1, max_in=prm_2, min_out=min_out,
                max_out=max_out))
    return new


def load_exp(flp, msg=None):
    """
    Loads one experiment results file (as generated by gen_features.py).
    """
    print(f"{'' if msg is None else f'{msg} - '}Parsing: {flp}")
    exp = Exp(flp)
    try:
        with np.load(flp, allow_pickle=True) as fil:
            num_files = len(fil.files)
            # Make sure that the results have the correct number of flows.
            if num_files == exp.tot_flws:
                dat = [fil[flw] for flw in fil.files]
            else:
                print(
                    f"\tThe number of subfiles ({num_files}) does not match "
                    f"the number of flows ({exp.tot_flws}): {flp}")
                dat = None

    except zipfile.BadZipFile:
        print(f"Bad simulation file: {flp}")
        dat = None
    return exp, dat


def clean(arr):
    """
    "Cleans" the provided numpy array by removing its column names. I.e., this
    converts a structured numpy array into a regular numpy array. Assumes that
    dtypes can be converted to float. If the
    """
    assert arr.dtype.names is not None, \
        f"The provided array is not structured. dtype: {arr.dtype.descr}"
    num_dims = len(arr.shape)
    assert num_dims == 1, \
        ("Only 1D structured arrays are supported, but this one has "
         f"{num_dims} dims!")

    num_cols = len(arr.dtype.names)
    new = np.empty((arr.shape[0], num_cols), dtype=float)
    for col in range(num_cols):
        new[:, col] = arr[arr.dtype.names[col]]
    return new


def visualize_classes(net, dat):
    """
    Prints statistics about the classes in dat.

    dat may be a torch Tensor, a numpy ndarray, or a list of dataloaders.
    """
    if isinstance(dat, (tuple, list)):
        dat = torch.cat([ldr.dataset.dat_out for ldr in dat])
    clss = net.get_classes()
    tots = get_class_popularity(dat, clss)
    # The total number of class labels extracted in the previous line.
    tot = sum(tots)
    net.log("\n".join(
        [f"\t{cls}: {tot_cls} examples ({tot_cls / tot * 100:.2f}%)"
         for cls, tot_cls in zip(clss, tots)]))
    tot_actual = dat.size()[0] if isinstance(dat, torch.Tensor) else dat.size
    assert tot == tot_actual, \
        f"Error visualizing ground truth! {tot} != {tot_actual}"


def get_class_popularity(dat, classes):
    """ Returns a list containing the number of examples in each class. """
    # Handles the cases where dat is a torch tensor, numpy unstructured array,
    # or numpy structured array containing a column named "class".
    dat = (
        dat if isinstance(dat, torch.Tensor) or dat.dtype.names is None
        else dat[features.LABEL_FET])
    return [(dat == cls).sum() for cls in classes]


def safe_mathis_label(tput_true, tput_mathis):
    """
    Returns the Mathis model label based on the true throughput and
    Mathis model fair throughput. If either component value is -1
    (unknown), then the resulting label is -1 (unknown).
    """
    return (
        -1 if tput_true == -1 or tput_mathis == -1 else
        int(tput_true > tput_mathis))


def safe_min(val1, val2):
    """
    Safely computes the min of two values. If either value is -1 or 0,
    then that value is discarded and the other value becomes the
    min. If both values are discarded, then the min is -1 (unknown).
    """
    return (
        -1 if val1 in UNSAFE and val2 in UNSAFE else (
            val2 if val1 in UNSAFE else (
                val1 if val2 in UNSAFE else (
                    min(val1, val2)))))


def safe_add(val1, val2):
    """
    Safely adds two values. If either value is -1, then the
    result is -1 (unknown).
    """
    return -1 if val1 == -1 or val2 == -1 else val1 + val2


def safe_sub(val1, val2):
    """
    Safely subtracts two values. If either value is -1, then the
    result is -1 (unknown).
    """
    return -1 if val1 == -1 or val2 == -1 else val1 - val2


def safe_mul(val1, val2):
    """
    Safely multiplies two values. If either value is -1, then the
    result is -1 (unknown).
    """
    return -1 if val1 == -1 or val2 == -1 else val1 * val2


def safe_div(num, den):
    """
    Safely divides two values. If either value is -1 or the denominator is 0,
    then the result is -1 (unknown).
    """
    return -1 if num == -1 or den in UNSAFE else num / den


def safe_np_div(num_arr, den):
    """
    Safely divides a 1D numpy array by a scalar. If an entry in the numerator
    array is -1 (unknown), then that entry in the output array is -1. If the
    denominator scalar is -1, then all entries in the output array are -1.
    """
    assert num_arr.size == num_arr.shape[0], \
        f"Array is not 1D: {num_arr.shape}"

    out = np.full_like(num_arr, -1)
    if den == -1:
        return out
    # Popular known entries.
    mask = num_arr == -1
    out[mask] = num_arr[mask] / den
    return out


def safe_sqrt(val):
    """
    Safely calculates the square root of a value. If the value is -1 (unknown),
    then the result is -1 (unknown).
    """
    return -1 if val == -1 else math.sqrt(val)


def safe_abs(val):
    """
    Safely calculates the absolute value of a value. If the value is -1
    (unknown), then the result is -1 (unknown).
    """
    return -1 if val == -1 else abs(val)


def get_safe(dat, start_idx=None, end_idx=None):
    """
    Returns a filtered window between the two specified indices, with all
    unknown values (-1) removed.
    """
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = 0 if dat.shape[0] == 0 else dat.shape[0] - 1
    # Extract the window.
    dat_win = dat[start_idx:end_idx + 1]
    # Eliminate values that are -1 (unknown).
    return dat_win[dat_win != -1]


def safe_sum(dat, start_idx=None, end_idx=None):
    """
    Safely calculates a sum over a window. Any values that are -1
    (unknown) are discarded. The sum of an empty window is -1 (unknown).
    """
    dat_safe = get_safe(dat, start_idx, end_idx)
    # If the window is empty, then the mean is -1 (unknown).
    return -1 if dat_safe.shape[0] == 0 else np.sum(dat_safe)


def safe_mean(dat, start_idx=None, end_idx=None):
    """
    Safely calculates a mean over a window. Any values that are -1
    (unknown) are discarded. The mean of an empty window is -1
    (unknown).
    """
    dat_safe = get_safe(dat, start_idx, end_idx)
    # If the window is empty, then the mean is -1 (unknown).
    return -1 if dat_safe.shape[0] == 0 else np.mean(dat_safe)


def safe_update_ewma(prev_ewma, new_val, alpha):
    """
    Safely updates an exponentially weighted moving average. If the previous
    EWMA is -1 (unknown), then the new EWMA is assumed to be the unweighted new
    value. If the new value is unknown, then the EWMA does not change.
    """
    return (
        new_val if prev_ewma == -1
        else (
            prev_ewma if new_val == -1
            else alpha * new_val + (1 - alpha) * prev_ewma))


def filt(dat_in, dat_out, dat_extra, scl_grps, num_sims, prc):
    """
    Filters parsed data based on a desired number of simulations and percent of
    results from each simulation. Each dat_* is a Python list, where each entry
    is a Numpy array containing the results of one simulation.
    """
    assert (
        len(dat_in) >= num_sims and
        len(dat_out) == len(dat_in) and
        len(dat_extra) == len(dat_in)), \
        "Arguments contain the wrong number of experiments!"
    # Pick the desired number of simulations.
    dat_in = dat_in[:num_sims]
    dat_out = dat_out[:num_sims]
    dat_extra = dat_extra[:num_sims]
    scl_grps = scl_grps[:num_sims]
    # From each simulation, pick the desired fraction.
    if prc != 100:
        for idx in range(num_sims):
            num_rows = dat_in[idx].shape[0]
            idxs = np.random.random_integers(
                0, num_rows - 1, math.ceil(num_rows * prc / 100))
            dat_in[idx] = dat_in[idx][idxs]
            dat_out[idx] = dat_out[idx][idxs]
            dat_extra[idx] = dat_extra[idx][idxs]
    return dat_in, dat_out, dat_extra, scl_grps


def save_parsed_data(flp, trn, val, tst):
    """
    Saves parsed data. Each dat_* is a list where each entry is the results of
    one simulation.
    """
    print(f"Saving data: {flp}")
    np.savez_compressed(
        flp,
        train_in=trn[0],
        train_out=trn[1],
        train_extra=trn[2],
        val_in=val[0],
        val_out=val[1],
        val_extra=val[2],
        test_in=tst[0],
        test_out=tst[1],
        test_extra=tst[2])


def load_parsed_data(flp):
    """ Loads parsed data. """
    print(f"Loading data: {flp}")
    with np.load(flp) as fil:
        splits = [
            (fil[f"{split}_in"], fil[f"{split}_out"], fil[f"{split}_extra"])
            for split in ["train", "val", "test"]]

    # Make sure that each set of in, out, and extra matrices contains the same
    # number of rows.
    for num_lens in [{dat.shape[0] for dat in dats} for dats in splits]:
        assert len(num_lens) == 1, \
            f"Mismatched in, out, and extra dimensions: {flp}"
    return splits


def save_tmp_file(flp, dat_in, dat_out, dat_extra, scl_grps):
    """ Saves a single-simulation temporary results file. """
    print(f"Saving temporary data: {flp}")
    np.savez_compressed(
        flp, dat_in=dat_in, dat_out=dat_out, dat_extra=dat_extra,
        scl_grps=scl_grps)


def load_tmp_file(flp):
    """ Loads and deletes a single-experiment temporary results file. """
    print(f"Loading temporary data: {flp}")
    with np.load(flp) as fil:
        dat_in = fil["dat_in"]
        dat_out = fil["dat_out"]
        dat_extra = fil["dat_extra"]
        scl_grps = fil["scl_grps"]
    os.remove(flp)
    return dat_in, dat_out, dat_extra, scl_grps


def get_lock_flp(out_dir):
    """ Returns the path to a lock file in out_dir. """
    return path.join(out_dir, defaults.LOCK_FLN)


def create_lock_file(out_dir):
    """ Creates a lock file in out_dir. """
    lock_flp = get_lock_flp(out_dir)
    if not path.exists(lock_flp):
        with open(lock_flp, "w") as _:
            pass


def check_lock_file(out_dir):
    """ Checks whether a lock file exists in out_dir. """
    return path.exists(get_lock_flp(out_dir))


def remove_lock_file(out_dir):
    """ Remove a lock file from out_dir. """
    try:
        os.remove(get_lock_flp(out_dir))
    except FileNotFoundError:
        pass


def get_npz_headers(flp):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files, and
    returns a list of tuples of the form:
        (name, shape, np.dtype)

    Adapted from: https://stackoverflow.com/a/43223420
    """
    def decode_header(archive, name):
        """ Decodes the header information of a single NPY file. """
        npy = archive.open(name)
        version = np.lib.format.read_magic(npy)
        shape, _, dtype = np.lib.format._read_array_header(npy, version)
        return [name[:-4], shape, dtype]

    with zipfile.ZipFile(flp) as archive:
        return [
            decode_header(archive, name) for name in archive.namelist()
            if name.endswith(".npy")]


def set_rand_seed(seed=defaults.SEED):
    """ Sets the Python, numpy, and Torch random seeds to seed. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_tensor(**kwargs):
    """
    Asserts that the provided values (given as keyword args) are Torch tensors.
    """
    for name, val in kwargs.items():
        assert isinstance(val, torch.Tensor), \
            (f"\"{name}\" is of type \"{type(val)}\" when it should be of type "
             "\"torch.Tensor\"")


def has_non_finite(arr):
    """ Returns whether the provided array contains any NaNs or Infs. """
    for fet in arr.dtype.names:
        if not np.isfinite(arr[fet]).all():
            return True
    return False


def bdp_B(bw_Mbps, rtt_us):
    """ Calculates the BDP in bytes. """
    return (bw_Mbps / 8. * 1e6) * (rtt_us / 1e6)


def get_split_data_flp(split_dir, name):
    """
    Returns the path to the data for a Split with the provided name, which
    stores its data in the provided directory.
    """
    return path.join(split_dir, f"{name}.npy")


def get_split_metadata_flp(split_dir, name):
    """
    Returns the path to the metadata for a Split with the provided name, which
    stores its data in the provided directory.
    """
    return path.join(split_dir, f"{name}_metadata.pickle")


def save_split_metadata(split_dir, name, dat):
    """
    Saves the metadata associated with a Split with the provided name that
    stores its data in the provided directory.
    """
    with open(get_split_metadata_flp(split_dir, name), "wb") as fil:
        pickle.dump(dat, fil)


def load_split_metadata(split_dir, name):
    """
    Loads metadata for a Split with the provided name that stores its data in
    the provided directory.
    """
    with open(get_split_metadata_flp(split_dir, name), "rb") as fil:
        return pickle.load(fil)


def get_scl_prms_flp(out_dir):
    """
    Returns the path to a scaling parameters file in the provided directoy.
    """
    return path.join(out_dir, "scale_params.json")


def save_scl_prms(out_dir, scl_prms):
    """ Saves scaling parameters in the provided directory. """
    scl_prms_flp = get_scl_prms_flp(out_dir)
    print(f"Saving scaling parameters: {scl_prms_flp}")
    with open(scl_prms_flp, "w") as fil:
        json.dump(scl_prms.tolist(), fil)


def load_scl_prms(out_dir):
    """ Loads scaling parameters from the provided directory. """
    with open(get_scl_prms_flp(out_dir), "r") as fil:
        return json.load(fil)


def load_split(split_dir, name):
    """
    Loads a training, validation, and test split's raw data from disk and
    returns it.
    """
    num_pkts, dtype = load_split_metadata(split_dir, name)
    if num_pkts == 0:
        # If the number of packets in this split is 0, then we will not find the
        # split on disk (because it is impossible to create a memory-mapped
        # numpy ndarray of size 0). Therefore, just return a new empty numpy
        # ndarray.
        return np.zeros((num_pkts,), dtype=dtype)
    return np.memmap(
        get_split_data_flp(split_dir, name), dtype=dtype, mode="r",
        shape=(num_pkts,))


def load_subsplits(split_dir, prefix):
    """ Loads and returns splits that begin with prefix. """
    subsplits = [
        load_split(split_dir, fil.split("_metadata.")[0])
        for fil in os.listdir(split_dir)
        if fil.startswith(prefix) and fil.endswith(".pickle")]
    assert subsplits, \
        f"No subsplits found with prefix \"{prefix}\" in: {split_dir}"
    return subsplits


def get_feature_analysis_flp(out_dir):
    """
    Returns the path to the feature analysis log file in the provided directory.
    """
    return path.join(out_dir, "feature_analysis.txt")


def log_feature_analysis(out_dir, msg):
    """
    Prints a feature analysis log statement while also writing it to a file.
    """
    print(msg)
    with open(get_feature_analysis_flp(out_dir), "a+") as fil:
        fil.write(msg)


def analyze_feature_correlation(net, out_dir, dat_in, clusters):
    """ Analyzes correlation among features of a net. """
    assert_tensor(dat_in=dat_in)
    # Convert all unknown values (-1) and NaNs to the mean of their column
    # because calculating the correlation using unknown values does not make
    # sense and stats.spearmanr() does not like NaNs, respectively.
    for col in range(dat_in.size()[1]):
        invalid = torch.logical_or(dat_in[:,col].isnan(), dat_in[:,col] == -1)
        dat_in[:,col][torch.nonzero(invalid)] = torch.mean(
            dat_in[:,col][torch.nonzero(torch.logical_not(invalid))])

    # Feature analysis.
    fets = np.asarray(net.in_spc)
    corr = stats.spearmanr(dat_in).correlation
    # corr = stats.pearsonr(dat_in_cleaned, dat_out_cleaned).correlation
    # corr = np.corrcoef(dat_in_cleaned, dat_out_cleaned, rowvar=False)
    corr_linkage = cluster.hierarchy.ward(corr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
    # Dendrogram.
    dendro = cluster.hierarchy.dendrogram(
        corr_linkage, labels=fets, ax=ax1, leaf_rotation=90)
    # Heatmap.
    dendro_idx = np.arange(0, len(dendro['ivl']))
    heatmap = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical', fontsize=5)
    ax2.set_yticklabels(dendro['ivl'], fontsize=5)
    # Create colorbar.
    ax2.figure.colorbar(heatmap, ax=ax2)
    fig.tight_layout()
    plt.savefig(path.join(out_dir, f"dendrogram_{net.name}.pdf"))

    # Determine which cluster each feature belongs to. Find a cluster threshold
    # that yields the desired number of clusters.
    cluster_thresh = 1
    attempts = 0
    while attempts < defaults.CLUSTER_ATTEMPTS:
        attempts += 1
        # Maps cluster index to a list of the indices of features in that cluster.
        cluster_to_fets = collections.defaultdict(list)
        for feature_idx, cluster_idx in enumerate(cluster.hierarchy.fcluster(
                corr_linkage, cluster_thresh, criterion='distance')):
            cluster_to_fets[cluster_idx].append(fets[feature_idx])
        num_clusters = len(cluster_to_fets)
        if num_clusters == clusters:
            # Found the desired number of clusters.
            break
        if num_clusters > clusters:
            # Too many clusters. Raise the threshold.
            cluster_thresh *= 1.01
        else:
            # Too fwe clusters. Lower the threshold.
            cluster_thresh -= 0.01
    else:
        raise Exception(
            f"Unable to find a suitable cluster threshold for {clusters} "
            f"clusters after {defaults.CLUSTER_ATTEMPTS} attempts.")

    # Print the clusters.
    log_feature_analysis(
        out_dir,
        f"Feature clusters ({len(cluster_to_fets)}):\n" + "\n".join(
            (f"\t{cluster_id}:\n\t\t" + "\n\t\t".join(fets))
            for cluster_id, fets in sorted(cluster_to_fets.items())))
    # Print the first feature in every cluster.
    log_feature_analysis(
        out_dir,
        "Naively-selected features: " +
        ", ".join(cluster_fets[0] for cluster_fets in cluster_to_fets.values()))

    return cluster_to_fets


def analyze_feature_importance(net, out_dir, dat_in, dat_out,
                               num_fets_to_pick, perm_imp_repeats):
    """ Analyzes the importance of features to a trained net. """
    # Analyze feature coefficients. The underlying model's .coef_
    # attribute may not exist.
    print("Analyzing feature importances...")
    tim_srt_s = time.time()
    fets = net.in_spc
    try:
        if isinstance(
                net.net,
                (feature_selection.RFE,
                 feature_selection.RFECV)):
            # Since the model was trained using RFE, display all
            # features. Sort the features alphabetically.
            top_fets = sorted(
                zip(np.array(fets)[(net.net.ranking_ == 1).nonzero()],
                    net.net.estimator_.coef_[0]),
                key=lambda p: p[0])
            log_feature_analysis(
                out_dir, f"Number of features selected: {len(top_fets)}")
            qualifier = "All"
        else:
            qualifier = "Best"
            if isinstance(
                    net.net, ensemble.HistGradientBoostingClassifier):
                imps = inspection.permutation_importance(
                    net.net, dat_in, dat_out, n_repeats=perm_imp_repeats,
                    random_state=0).importances_mean
            else:
                imps = net.net.coef_[0]

            # First, sort the features by the absolute value of the
            # importance and pick the top 20. Then, sort the features
            # alphabetically.
            # top_fets = sorted(
            #     sorted(
            #         zip(fets, imps),
            #         key=lambda p: abs(p[1]))[-20:],
            #     key=lambda p: p[0])
            top_fets = list(reversed(sorted(
                zip(fets, imps), key=lambda p: abs(p[1]))))
            if num_fets_to_pick is not None:
                top_fets = top_fets[:num_fets_to_pick]

        log_feature_analysis(
            out_dir,
            f"----------\n{qualifier} features ({len(top_fets)}):\n" +
            "\n".join(f"\t{fet}: {coef:.4f}" for fet, coef in top_fets) +
            "\n----------")

        # Graph feature coefficients.
        if net.graph:
            names, coefs = zip(*top_fets)
            num_fets = len(names)
            y_vals = list(range(num_fets))
            plt.figure(figsize=(7, 0.2 * num_fets))
            plt.barh(y_vals, coefs, align="center")
            plt.yticks(y_vals, names)
            plt.ylim((-1, num_fets))
            plt.xlabel("Feature coefficient")
            plt.ylabel("Feature name")
            plt.tight_layout()
            plt.savefig(path.join(out_dir, f"features_{net.name}.pdf"))
            plt.close()

        return top_fets
    except AttributeError:
        # Coefficients are only available with a linear kernel.
        log_feature_analysis(
            out_dir, "Warning: Unable to extract coefficients!")
    finally:
        log_feature_analysis(
            out_dir,
            ("Finished analyzing feature importance - time: "
             f"{time.time()- tim_srt_s:.2f} seconds"))
    return None


def check_fets(fets, in_spc):
    """
    Verifies that the processed features that emerge from the data processing
    pipeline are the same as a net's in_spc. fets is intended to come from the
    data processing pipeline (i.e., the actual features after all data
    processing). in_spc is intended to come from a net (i.e., the net's original
    feature specification).
    """
    assert fets == in_spc, \
        ("Provided features do not agreed with in_spc."
        f"\n\tProvided fets ({len(fets)}): {fets}"
         f"\n\tin_spc ({len(in_spc)}): {in_spc}")


def zip_timeseries(xs, ys):
    """ Zips together multiple timeseries from the same timespace. """
    assert xs
    assert ys
    assert len(xs) == len(ys)
    for idx in range(len(xs)):
        assert xs[idx].shape[0] == ys[idx].shape[0]

    idxs = [0] * len(xs)
    tot = sum(xs_.shape[0] for xs_ in xs)
    xs_o = np.full((tot,), -1, dtype=xs[0].dtype)
    ys_o = np.full((tot,), -1, dtype=ys[0].dtype)
    idx_o = 0

    while idx_o < xs_o.shape[0]:
        chosen = None
        earliest = sys.maxsize

        for idx in range(len(xs)):
            if idxs[idx] < xs[idx].shape[0]:
                proposed_earliest = xs[idx][idxs[idx]]
                if proposed_earliest < earliest:
                    chosen = idx
                    earliest = proposed_earliest
        assert chosen is not None, "Ran out of points."

        xs_o[idx_o] = xs[chosen][idxs[chosen]]
        ys_o[idx_o] = ys[chosen][idxs[chosen]]
        idx_o += 1
        idxs[chosen] += 1
    return xs_o, ys_o

def select_fets(cluster_to_fets, top_fets):
    """
    Selects the most important feature from each cluster in cluster_to_fets,
    using the importance information in top_fets. Each entry in top_fets is a
    tuple of the form: (feature name, feature importance).
    """
    # Returns the dictionary keys whose values contain an item.
    get_keys = lambda x, d: [k for k, v in d.items() if x in v]
    chosen_fets = []
    # Examine the features from most important to least important.
    for fet in reversed(sorted(top_fets, key=lambda p: p[1])):
        fet_name = fet[0]
        clusters = get_keys(fet_name, cluster_to_fets)
        assert len(clusters) <= 1, \
            ("A feature should be in either 0 or 1 clusters, but "
             f"\"{fet_name}\" is in clusters: {clusters}")
        if len(clusters) == 0:
            # This feature's cluster has already been used.
            continue
        # This is the first features from this cluster to be used, so
        # keep it.
        chosen_fets.append(fet)
        # Remove this cluster to invalidate its other features.
        del cluster_to_fets[clusters[0]]
    # Make sure that chosen features are sorted in decreasing order of
    # importance (most important is first).
    chosen_fets = list(reversed(sorted(chosen_fets, key=lambda p:p[1])))
    print(
        f"Chosen features ({len(chosen_fets)}):\n\t" +
        "\n\t".join(
            f"{fet}: {coeff:.4f}" for fet, coeff in chosen_fets))
    print(
        "New in_spc:", "\tin_spc = (",
        "\t\t" + "\n\t\t".join(f"\"{fet}\"," for fet, _ in chosen_fets[:10]),
        "\t)", sep="\n")
    return chosen_fets


def find_bound(vals, target, min_idx, max_idx, which):
    """
    Returns the first index that is either before or after a particular target.
    vals must be monotonically increasing.
    """
    assert min_idx >= 0
    assert max_idx >= min_idx
    assert which in {"before", "after"}
    if min_idx == max_idx:
        return min_idx

    bound = min_idx
    # Walk forward until the target time is in the past.
    while bound < (max_idx if which == "before" else max_idx - 1):
        time_us = vals[bound]
        if time_us == -1 or time_us < target:
            bound += 1
        else:
            break

    if which == "before":
        # If we walked forward, then walk backward to the last valid time.
        while bound > min_idx:
            bound -= 1
            if vals[bound] != -1:
                break

    assert min_idx <= bound <= max_idx
    return bound
