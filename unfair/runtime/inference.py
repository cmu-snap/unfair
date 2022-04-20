"""This module defines a process that will receive packets and run inference on them."""

import collections
import ctypes
import logging
import os
from os import path
import queue
import signal
import time
import traceback

from bcc import BPF, BPFAttachType
import numpy as np
from pyroute2 import IPRoute, protocols
from pyroute2.netlink.exceptions import NetlinkError

from unfair.model import data, defaults, features, gen_features, models, utils
from unfair.runtime import flow_utils, reaction_strategy
from unfair.runtime.reaction_strategy import ReactionStrategy


def inference(net, flowkey, min_rtt_us, fets, prev_fets=None, debug=False):
    """Run inference on a flow's packets.

    Returns a label (below fair, approximately fair, above fair), the updated
    min_rtt_us, and the features of the last packet.
    """
    gen_features.parse_received_acks(flowkey, min_rtt_us, fets, prev_fets)

    # Remove unneeded features that were added as dependencies for the requested
    # features. Only run prediction on the last packet.
    in_fets = fets[-1:][list(net.in_spc)]
    # Replace -1's and with NaNs and convert to an unstructured numpy array.
    data.replace_unknowns(in_fets, isinstance(net, models.HistGbdtSklearnWrapper))
    in_fets = utils.clean(in_fets)

    if debug:
        logging.debug(
            "Model input: %s\n%s",
            net.in_spc,
            "\n".join(", ".join(f"{fet}" for fet in row) for row in in_fets),
        )

    pred_start_s = time.time()
    preds = net.predict(in_fets)
    logging.info("Prediction time: %.2f ms", (time.time() - pred_start_s) * 1e3)

    return [defaults.Class(pred) for pred in preds]


def condense_labels(labels):
    """Combine multiple labels into a single label.

    For example, smooth the labels by selecting the average label.

    Currently, this simply selects the last label.
    """
    assert len(labels) > 0, "Labels cannot be empty."
    return labels[-1]


def make_decision(
    args, flowkey, min_rtt_us, fets, label, flow_to_decisions, flow_to_rwnd
):
    """Make a flow unfairness mitigation decision.

    Base the decision on the flow's label and existing decision. Use the flow's features
    to calculate any necessary flow metrics, such as the throughput.
    """
    logging.info("Label for flow %s: %s", flowkey, label)

    logging.info("Num decisions: %d", len(flow_to_decisions))
    logging.info("Num rwnds: %d", len(flow_to_rwnd))

    if args.reaction_strategy == ReactionStrategy.FILE:
        new_decision = (
            defaults.Decision.PACED,
            None,
            reaction_strategy.get_scheduled_pacing(args.schedule),
        )
    else:
        tput_bps = utils.safe_tput_bps(fets, 0, len(fets) - 1)
        if label == defaults.Class.ABOVE_FAIR:
            # if flow_to_decisions[flowkey][0] == defaults.Decision.PACED:
            #     logging.info("existing_tput_bps: %f", flow_to_decisions[flowkey][1])
            # This flow is sending too fast. Force the sender to slow down.
            new_tput_bps = reaction_strategy.react_down(
                args.reaction_strategy,
                # If the flow was already paced, then based the new paced throughput on
                # the old paced throughput.
                flow_to_decisions[flowkey][1]
                if flow_to_decisions[flowkey][0] == defaults.Decision.PACED
                else tput_bps,
            )
            # logging.info(
            #     "new_tput_bps: %f, current tput_bps: %f", new_tput_bps, tput_bps
            # )
            new_decision = (
                defaults.Decision.PACED,
                new_tput_bps,
                utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
            )
        elif flow_to_decisions[flowkey][0] == defaults.Decision.PACED:
            # We are already pacing this flow.
            if label == defaults.Class.BELOW_FAIR:
                # If we are already pacing this flow but we are being too
                # aggressive, then let it send faster.
                new_tput_bps = reaction_strategy.react_up(
                    args.reaction_strategy,
                    # If the flow was already paced, then based the new paced throughput on
                    # the old paced throughput.
                    flow_to_decisions[flowkey][1]
                    if flow_to_decisions[flowkey][0] == defaults.Decision.PACED
                    else tput_bps,
                )
                new_decision = (
                    defaults.Decision.PACED,
                    new_tput_bps,
                    utils.bdp_B(new_tput_bps, min_rtt_us / 1e6),
                )
            else:
                # If we are already pacing this flow and it is behaving as desired,
                # then all is well. Retain the existing pacing decision.
                new_decision = flow_to_decisions[flowkey]
        else:
            # This flow is not already being paced and is not behaving unfairly, so
            # leave it alone.
            new_decision = (defaults.Decision.NOT_PACED, None, None)

    # FIXME: Why are the BDP calculations coming out so small? Is the throughput
    #        just low due to low application demand?

    logging.info(
        "Decision for flow %s: (%s, target tput: %s, rwnd: %s)",
        flowkey,
        new_decision[0],
        "-" if new_decision[1] is None else f"{new_decision[1] / 1e6:.2f} Mbps",
        "-" if new_decision[2] is None else f"{new_decision[2] / 1e3:.2f} KB",
    )
    if flow_to_decisions[flowkey] != new_decision:
        logging.info("Flow %s changed decision.", flowkey)
        if new_decision[2] is None:
            if flowkey in flow_to_rwnd:
                del flow_to_rwnd[flowkey]
        else:
            new_decision = (new_decision[0], new_decision[1], round(new_decision[2]))
            # if new_decision[2] > 2**16:
            #     logging.info(f"Warning: Asking for RWND >= 2**16: {new_decision[2]}")
            #     new_decision[2] = 2**16 - 1
            if new_decision[2] < defaults.MIN_RWND_B:
                logging.info(
                    ("Warning: Flow %s asking for RWND < %d: %d. " "Overriding to %d."),
                    flowkey,
                    defaults.MIN_RWND_B,
                    new_decision[2],
                    defaults.MIN_RWND_B,
                )
                new_decision = (new_decision[0], new_decision[1], defaults.MIN_RWND_B)

            assert new_decision[2] >= 0, (
                "Error: RWND must be non-negative, "
                f"but is {new_decision[2]} for flow {flowkey}."
            )
            flow_to_rwnd[flowkey] = ctypes.c_uint32(new_decision[2])
        flow_to_decisions[flowkey] = new_decision


def packets_to_ndarray(pkts, dtype):
    """Reorganize a list of packet metrics into a structured numpy array."""
    # Assume that the packets are in order.
    # # For some reason, the packets tend to get reordered after they are timestamped on
    # # arrival. Sort packets by timestamp.
    # pkts = sorted(pkts, key=lambda pkt: pkt[-1])
    (
        seqs,
        rtts_us,
        # tsvals,
        # tsecrs,
        totals_bytes,
        # _,
        # _,
        payloads_bytes,
        times_us,
    ) = zip(*pkts)
    # The final features. -1 implies that a value could not be calculated. Extend the
    # provided dtype with the regular features, which may be required to compute the
    # EWMA and windowed features.
    fets = np.full(len(seqs), -1, dtype=dtype)
    fets[features.SEQ_FET] = seqs
    fets[features.ARRIVAL_TIME_FET] = times_us
    fets[features.PAYLOAD_FET] = payloads_bytes
    fets[features.WIRELEN_FET] = totals_bytes
    fets[features.RTT_FET] = rtts_us
    return fets


def load_bpf():
    """Load the corresponding eBPF program."""
    # Load BPF text.
    bpf_flp = path.join(
        path.abspath(path.dirname(__file__)),
        "unfair_runtime.c",
    )
    if not path.isfile(bpf_flp):
        logging.error("Could not find BPF program: %s", bpf_flp)
        return 1
    logging.info("Loading BPF program: %s", bpf_flp)
    with open(bpf_flp, "r", encoding="utf-8") as fil:
        bpf_text = fil.read()
    # Load BPF program.
    return BPF(text=bpf_text)


def configure_ebpf(args):
    """Set up eBPF hooks."""
    bpf = load_bpf()

    # Read the TCP window scale on outgoing SYN-ACK packets.
    func_sock_ops = bpf.load_func("read_win_scale", bpf.SOCK_OPS)  # sock_stuff
    filedesc = os.open(args.cgroup, os.O_RDONLY)
    bpf.attach_func(func_sock_ops, filedesc, BPFAttachType.CGROUP_SOCK_OPS)

    # Overwrite advertised window size in outgoing packets.
    egress_fn = bpf.load_func("handle_egress", BPF.SCHED_ACT)
    flow_to_rwnd = bpf["flow_to_rwnd"]
    # Set up a TC egress qdisc, specify a filter the accepts all packets, and attach
    # our egress function as the action on that filter.
    ipr = IPRoute()
    ifindex = ipr.link_lookup(ifname=args.interface)
    assert (
        len(ifindex) == 1
    ), f'Trouble looking up index for interface "{args.interface}": {ifindex}'
    ifindex = ifindex[0]
    # ipr.tc("add", "pfifo", 0, "1:")
    # ipr.tc("add-filter", "bpf", 0, ":1", fd=egress_fn.fd, name=egress_fn.name, parent="1:")
    action = dict(kind="bpf", fd=egress_fn.fd, name=egress_fn.name, action="ok")
    try:
        # Add the action to a u32 match-all filter
        ipr.tc("add", "htb", ifindex, 0x10000, default=0x200000)
        ipr.tc(
            "add-filter",
            "u32",
            ifindex,
            parent=0x10000,
            prio=10,
            protocol=protocols.ETH_P_ALL,  # Every packet
            target=0x10020,
            keys=["0x0/0x0+0"],
            action=action,
        )
    except NetlinkError:
        logging.error("Error: Unable to configure TC.")
        return None, None

    def ebpf_cleanup():
        """Clean attached eBPF programs."""
        logging.info("Detaching sock_ops hook...")
        bpf.detach_func(func_sock_ops, filedesc, BPFAttachType.CGROUP_SOCK_OPS)

        logging.info("Removing egress TC...")
        ipr.tc("del", "htb", ifindex, 0x10000, default=0x200000)

    return flow_to_rwnd, ebpf_cleanup


def inference_loop(args, flow_to_rwnd, que, inference_flags, done):
    """Receive packets and run inference on them."""
    net = models.load_model(args.model_file)

    flow_to_prev_features = {}
    flow_to_decisions = collections.defaultdict(
        lambda: (defaults.Decision.NOT_PACED, None, None)
    )

    # The final features. -1 implies that a value could not be calculated. Extend the
    # provided dtype with the regular features, which may be required to compute the
    # EWMA and windowed features.
    dtype = sorted(
        list(
            set(features.PARSE_PACKETS_FETS)
            | set(
                features.feature_names_to_dtype(features.fill_dependencies(net.in_spc))
            )
        )
    )

    logging.info("Inference ready!")
    while not done.is_set():
        try:
            val = que.get(timeout=1)
            # For some reason, the queue returns True or None when the thread on the
            # other end dies.
            if not isinstance(val, tuple):
                continue
        except queue.Empty:
            continue

        opcode, fourtuple = val[:2]
        flowkey = flow_utils.FlowKey(*fourtuple)

        if opcode == "inference":
            pkts, min_rtt_us = val[2:]
        elif opcode == "remove":
            logging.info("Inference process: Removing flow %s", flowkey)
            if flowkey in flow_to_rwnd:
                del flow_to_rwnd[flowkey]
            if flowkey in flow_to_decisions:
                del flow_to_decisions[flowkey]
            if flowkey in flow_to_prev_features:
                del flow_to_prev_features[flowkey]
            continue
        else:
            raise RuntimeError(f'Unknown opcode "{opcode}" for flow: {flowkey}')

        start_time_s = time.time()
        fets = packets_to_ndarray(pkts, dtype)
        try:
            labels = inference(
                net,
                flowkey,
                min_rtt_us,
                fets,
                flow_to_prev_features.get(flowkey),
                args.debug,
            )
            flow_to_prev_features[flowkey] = fets[-1]
        except AssertionError:
            # Assertion errors mean this batch of packets violated some precondition,
            # but we are safe to skip them and continue.
            logging.warning(
                "Inference failed due to a non-fatal assertion failure:\n%s",
                traceback.format_exc(),
            )
            continue
        except Exception as exp:
            # An unexpected error occurred. It is not safe to continue. Reraise the
            # exception to kill the process.
            logging.error(
                "Inference failed due to an unexpected error:\n%s",
                traceback.format_exc(),
            )
            raise exp
        else:
            # Inference succeeded.
            make_decision(
                args,
                flowkey,
                min_rtt_us,
                fets,
                condense_labels(labels),
                flow_to_decisions,
                flow_to_rwnd,
            )
        finally:
            dur_s = time.time() - start_time_s
            pps = len(pkts) / dur_s
            logging.info(
                "Inference performance: %.2f ms, %.2f pps, %.2f Mbps",
                dur_s * 1e3,
                pps,
                pps * 1514 * 8 / 1e6,
            )
            inference_flags[fourtuple].value = 0


def run(args, que, inference_flags, done):
    """Receive packets and run inference on them.

    This function is designed to be the target of a process.
    """

    def signal_handler(sig, frame):
        logging.info("Inference process: You pressed Ctrl+C!")
        done.set()

    signal.signal(signal.SIGINT, signal_handler)

    cleanup = None
    try:
        flow_to_rwnd, cleanup = configure_ebpf(args)
        if flow_to_rwnd is None:
            return
        inference_loop(args, flow_to_rwnd, que, inference_flags, done)
    except KeyboardInterrupt:
        logging.info("Inference process: You pressed Ctrl+C!")
        done.set()
    finally:
        if cleanup is not None:
            cleanup()
