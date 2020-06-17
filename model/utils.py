#! /usr/bin/env python3
""" Utility functions. """

from os import path

import numpy as np
import scapy.utils
import scapy.layers.l2
import scapy.layers.inet
import scapy.layers.ppp


class Sim():
    """ Describes the parameters of a simulation. """

    def __init__(self, sim):
        if "/" in sim:
            sim = path.basename(sim)
        self.name = sim
        toks = sim.split("-")
        if sim.endswith(".npz"):
            # 8Mbps-9000us-489p-1unfair-4other-9000,9000,9000,9000,9000us-1380B-80s-2rttW.npz
            toks = toks[:-1]
        # 8Mbps-9000us-489p-1unfair-4other-9000,9000,9000,9000,9000us-1380B-80s
        (bw_Mbps, btl_delay_us, queue_p, unfair_flws, other_flws, edge_delays,
         payload_B, dur_s) = toks

        # Link bandwidth (Mbps).
        self.bw_Mbps = float(bw_Mbps[:-4])
        # Bottleneck router delay (us).
        self.btl_delay_us = float(btl_delay_us[:-2])
        # Queue size (packets).
        self.queue_p = float(queue_p[:-1])
        # Number of unfair flows
        self.unfair_flws = int(unfair_flws[:-6])
        # Number of other flows
        self.other_flws = int(other_flws[:-5])
        # Edge delays
        self.edge_delays = [int(del_us) for del_us in edge_delays[:-2].split(",")]
        # Packet size (bytes)
        self.payload_B = float(payload_B[:-1])
        # Experiment duration (s).
        self.dur_s = float(dur_s[:-1])


def parse_packets_endpoint(flp, packet_size_B):
    """
    Takes in a file path and returns (seq, timestamp).
    """
    # Not using parse_time_us for efficiency purpose
    return [
        (scapy.layers.ppp.PPP(pkt_dat)[scapy.layers.inet.TCP].seq,
         pkt_mdat.sec * 1e6 + pkt_mdat.usec)
        for pkt_dat, pkt_mdat in scapy.utils.RawPcapReader(flp)
        # Ignore non-data packets.
        if pkt_mdat.wirelen >= packet_size_B
    ]


def parse_packets_router(flp, packet_size_B):
    """
    Takes in a file path and returns (sender, timestamp).
    """
    # Not using parse_time_us for efficiency purpose
    return [
        # Parse each packet as a PPP packet.
        (int(scapy.layers.ppp.PPP(pkt_dat)[
            scapy.layers.inet.IP].src.split(".")[2]),
         pkt_mdat.sec * 1e6 + pkt_mdat.usec)
        for pkt_dat, pkt_mdat in scapy.utils.RawPcapReader(flp)
        # Ignore non-data packets.
        if pkt_mdat.wirelen >= packet_size_B
    ]


def scale(x, min_in, max_in, min_out, max_out):
    """
    Scales x, which is from the range [min_in, max_in], to the range
    [min_out, max_out].
    """
    assert min_in != max_in, "Divide by zero!"
    return min_out + (x - min_in) * (max_out - min_out) / (max_in - min_in)


def load_sim(flp):
    """
    Loads one simulation results file (generated by parse_dumbbell.py). Returns
    a tuple of the form: (total number of flows, results matrix).
    """
    print(f"Parsing: {flp}")
    with np.load(flp) as fil:
        assert len(fil.files) == 1 and "1" in fil.files, \
            "More than one unfair flow detected!"
        return Sim(flp), fil["1"]


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
        (f"Only 1D structured arrays are supported, but this one has {num_dims} "
         "dims!")

    num_cols = len(arr.dtype.descr)
    new = np.empty((arr.shape[0], num_cols), dtype=float)
    for col in range(num_cols):
        new[:, col] = arr[arr.dtype.names[col]]
    return new
