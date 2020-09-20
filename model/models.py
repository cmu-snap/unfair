""" Models. """

import copy
import functools
import math
import os
from os import path
import random

from matplotlib import pyplot
import numpy as np
import sklearn
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import svm
import torch

import defaults
import utils


SMOOTHING_THRESHOLD = 0.4
SLIDING_WINDOW_NUM_RTT = 1


class PytorchModelWrapper:
    """ A wrapper class for PyTorch models. """

    # The name of this model.
    name = None
    # The specification of the input tensor format.
    in_spc = []
    # The specification of the output tensor format.
    out_spc = []
    # The number of output classes.
    num_clss = 0
    # The loss function to use during training.
    los_fnc = None
    # The optimizer to use during training.
    opt = None
    # Model-specific parameters. Each model may use these differently.
    params = []

    def __init__(self):
        self.net = None
        self.graph = False
        self.__check()

    def __check(self):
        """
        Verifies that this PytorchModel instance has been initialized properly.
        """
        assert self.name is not None, "Empty name!"
        assert self.in_spc, "Empty in_spc!"
        assert self.out_spc, "Empty out_spc!"
        assert self.num_clss > 0, "Invalid number of output classes!"
        if not isinstance(self, SvmSklearnWrapper):
            assert self.los_fnc is not None, "No loss function!"
            assert self.opt is not None, "No optimizer!"

    def init_hidden(self, batch_size):
        """
        If this model is an LSTM, then this method returns the initialized
        hidden state. Otherwise, returns None.
        """
        return torch.zeros(()), torch.zeros(())

    @staticmethod
    def convert_to_class(sim, dat_out):
        """ Converts real-valued feature values into classes. """
        return dat_out

    def get_classes(self):
        """ Returns a list of all possible class labels. """
        return list(range(self.num_clss))

    def modify_data(self, sim, dat_in, dat_out):
        """ Performs an arbitrary transformation on the data. """
        return dat_in, dat_out, list(range(len(dat_in.dtype.names)))

    def check_output(self, out, target):
        """
        Returns the number of examples from out that were classified correctly,
        according to target. out and target must be Torch tensors.
        """
        utils.assert_tensor(out=out, target=target)
        size_out = out.size()
        size_target = target.size()
        assert size_target
        assert size_out[0] == size_target[0], \
            ("Output and target have different batch sizes (first dimension): "
             f"{size_out} != {size_target}")
        # Transform the output into classes.
        out = self._check_output_helper(out)
        size_out = out.size()
        assert size_out == size_target, \
            f"Output and target sizes do not match: {size_out} != {size_target}"
        # eq(): Compare the outputs to the labels.
        # type(): Cast the resulting bools to ints.
        # sum(): Sum them up to get the total number of correct predictions.
        return out.eq(target).type(torch.int).sum().item()

    def _check_output_helper(self, out):
        """
        Convert the raw network output into classes. out must be a torch Tensor.
        """
        utils.assert_tensor(out=out)
        # Assume a one-hot encoding of class probabilities. The class
        # is the index of the output entry with greatest value (i.e.,
        # highest probability). Set dim=1 because the first dimension
        # is the batch.
        size_out = out.size()
        assert size_out[1] == self.num_clss, \
            (f"Expecting one-hot encoding for {self.num_clss} classes, but "
             f"found size: {size_out}")
        return torch.argmax(out, dim=1)

    def new(self):
        """ Returns a new instance of the underlying torch.nn.Module. """
        raise Exception(
            ("Attempting to call \"new()\" on the PytorchModelWrapper base "
             "class itself."))


class BinaryModelWrapper(PytorchModelWrapper):
    """ A base class for binary classification models. """

    num_clss = 2

    def __init__(self, win=100, rtt_buckets=False, windows=False):
        super(BinaryModelWrapper, self).__init__()

        self.win = win
        self.rtt_buckets = rtt_buckets
        self.windows = windows
        # Determine layer dimensions. If we are bucketizing packets
        # based on arrival time, then there will be one input feature
        # for every bucket (self.win) plus one input feature for every
        # entry in self.in_spc. If we are not bucketizing packets,
        # then there will be one input feature for each entry in
        # self.in_spc for each packet (self.win).
        self.num_ins = (
            self.win + len(self.in_spc) if self.rtt_buckets else (
                len(self.in_spc) * self.win if self.windows else
                len(self.in_spc)))

    @staticmethod
    def convert_to_class(sim, dat_out):
        # Verify that the output features consist of exactly one column.
        assert len(dat_out.dtype.names) == 1, "Should be only one column."
        # Map a conversion function across all entries. Note that
        # here an entry is an entire row, since each row is a
        # single tuple value.
        clss = np.vectorize(
            functools.partial(
                # Compare each queue occupancy percent with the fair
                # percent. prc[0] assumes a single column.
                lambda prc, fair: prc[0] > fair,
                fair=1. / (sim.unfair_flws + sim.fair_flws)),
            # Convert to integers.
            otypes=[int])(dat_out)
        clss_str = np.empty((clss.shape[0],), dtype=[("class", "int")])
        clss_str["class"] = clss
        return clss_str

    @staticmethod
    def __bucketize(dat_in, dat_extra, dat_in_start_idx, dat_in_end_idx,
                    dat_in_new, dat_in_new_idx, dur_us, num_buckets):
        """
        Uses dat_extra["arrival time us"] to divide the arriving packets from
        the range [dat_in_start_idx, dat_in_end_idx] into num_buckets
        intervals. Each interval has duration dur_us / num_buckets.
        Stores the resulting histogram in dat_in_new, at the row
        indicated by dat_in_new_idx. Returns nothing.
        """
        arr_times = dat_extra[
            "arrival time us"][dat_in_start_idx:dat_in_end_idx + 1]
        num_pkts = arr_times.shape[0]
        assert num_pkts > 0, "Need more than 0 packets!"

        # The duration of each interval.
        interval_us = dur_us / num_buckets
        # The arrival time of the first packet, and therefore the
        # start of the first interval.
        start_time_us = arr_times[0]
        # Convert the arrival times to interval indices and loop over them.
        for interval_idx in np.floor(
                (arr_times - start_time_us) / interval_us).astype(int):
            if interval_idx == num_buckets:
                print(f"Warning: Interval is {interval_idx} when it should be "
                      f"in the range [0, {num_buckets}]. Fixing interval...")
                interval_idx -= 1
            assert 0 <= interval_idx < num_buckets, \
                (f"Invalid idx ({interval_idx}) for the number of buckets "
                 f"({num_buckets})!")
            dat_in_new[dat_in_new_idx][interval_idx] += 1
        # Set the values of the other features based on the last packet in this
        # window.
        for fet in dat_in.dtype.names:
            dat_in_new[fet][dat_in_new_idx] = dat_in[fet][dat_in_end_idx]

        # Check that the bucket features reflect all of the packets.
        bucketed_pkts = sum(dat_in_new[dat_in_new_idx].tolist()[:num_buckets])
        assert bucketed_pkts == num_pkts, \
            (f"Error building counts! Bucketed {bucketed_pkts} of {num_pkts} "
             "packets!")

    def __create_buckets(self, sim, dat_in, dat_out, dat_extra, sequential):
        """
        Divides dat_in into windows and divides each window into self.win
        buckets, which each defines a temporal interval. The value of
        each bucket is the number of packets that arrived during that
        interval. The output value for each window is the output of
        the last packet in the window.
        """
        print("Creating arrival time buckets...")
        # 100x the min RTT (as determined by the simulation parameters).
        dur_us = 100 * 2 * (sim.edge_delays[0] * 2 + sim.btl_delay_us)
        # Determine the safe starting index. Do not pick indices
        # between 0 and start_idx to make sure that all windows ending
        # on the chosen index fit within the simulation.
        first_arr_time = None
        start_idx = None
        for idx, arr_time in enumerate(dat_extra[defaults.ARRIVAL_TIME_FET]):
            if first_arr_time is None:
                first_arr_time = arr_time
                continue
            if arr_time - first_arr_time >= dur_us:
                start_idx = idx
                break
        num_pkts = dat_in.shape[0]
        assert start_idx is not None and 0 <= start_idx < num_pkts, \
            f"Invalid start index: {start_idx}"

        if sequential:
            # Select all valid windows, in order.
            num_wins = num_pkts - start_idx
            pkt_idxs = list(range(start_idx, num_pkts))
        else:
            # The number of windows is the number of times the window
            # durations fits within the simulation.
            num_wins = 10 * math.floor(sim.dur_s * 1e6 / dur_us)
            # Select random intervals from this simulation to create the
            # new input data. win_idx is the index into
            # dat_in_new. pkt_idx is the index of the last packet in this
            # window.
            pkt_idxs = random.choices(range(start_idx, num_pkts), k=num_wins)
        # Records the number of packets that arrived during each
        # interval. This is a structured numpy array where each column
        # is named based on its bucket index. We add extra columns for
        # the normal time features.
        dat_in_new = np.zeros(
            (num_wins,),
            dtype=([(f"bucket_{bkt}", "float") for bkt in range(self.win)] +
                   [col for col in dat_in.dtype.descr if col[0] != ""]))

        for win_idx, pkt_idx in enumerate(pkt_idxs):
            # Find the first packet in this window (whose index will
            # be win_start_idx).
            cur_arr_time = dat_extra[defaults.ARRIVAL_TIME_FET][pkt_idx]
            win_start_idx = None
            # Move backwards from the last packet in the window until
            # we find a packet whose arrival time is more that dur_us
            # in the past.
            for arr_time_idx in range(pkt_idx, -1, -1):
                if (cur_arr_time -
                        dat_extra[defaults.ARRIVAL_TIME_FET][arr_time_idx] >
                        dur_us):
                    # This packet is the first packet that is too far
                    # in the past, so we will start the window on the
                    # next packet.
                    win_start_idx = arr_time_idx + 1
                    break
            assert (
                win_start_idx is not None and 0 <= win_start_idx <= pkt_idx), \
                ("Problem finding beginning of window! Are there insufficient "
                 "packets?")
            self.__bucketize(
                dat_in, dat_extra, win_start_idx, pkt_idx, dat_in_new, win_idx,
                dur_us, self.win)

        # Verify that we selected at least as many windows as we intended to.
        num_selected_wins = len(dat_in_new)
        assert num_selected_wins >= num_wins, \
            f"Insufficient windows: {num_selected_wins} < {num_wins}"

        return (
            dat_in_new,
            # As an output feature, select only the final ground truth
            # value. I.e., the final ground truth value for this window
            # becomes the ground truth for the entire window.
            np.take(dat_out, pkt_idxs),
            np.take(dat_extra, pkt_idxs),
            # The buckets all share a scaling group. Each other
            # feature is part of its own group.
            [0] * self.win +
            list(range(1, len(dat_in_new.dtype.names) - self.win + 1)))

    def __create_windows(self, dat_in, dat_out, sequential):
        """
        Divides dat_in into windows of self.win packets. Flattens the
        features of the packets in a window. The output value for each
        window is the output of the last packet in the window.
        """
        print("Creating windows...")
        num_pkts = dat_in.shape[0]
        num_wins = math.ceil(num_pkts / self.win)
        fets = [(name, typ) for name, typ in dat_in.dtype.descr if name != ""]
        # Select random intervals from this simulation to create the
        # new input data. Do not pick indices between 0 and self.win
        # to make sure that all windows ending on the chosen index fit
        # within the simulation.
        pkt_idxs = random.choices(range(self.win, num_pkts), k=num_wins)
        # The new data format consists of self.win copies of the
        # existing input features. All copies of a particular feature
        # share the same scaling group.
        scl_grps, dtype = zip(
            *[(scl_grp, (f"{name}_{idx}", typ))
              for idx in range(self.win)
              for scl_grp, (name, typ) in enumerate(fets)])
        scl_grps = np.array(scl_grps)
        dat_in_new = np.zeros((num_wins,), dtype=list(dtype))

        for win_idx, end_idx in enumerate(pkt_idxs):
            # This could be done on a single line with a range select
            # and a generator, but this version is preferable because
            # it removes intermediate data copies and guarantees that
            # the resulting row is properly ordered.
            for fet_idx, pkt_idx in enumerate(
                    range(end_idx - self.win + 1, end_idx + 1)):
                for name, _ in fets:
                    dat_in_new[f"{name}_{fet_idx}"][win_idx] = (
                        dat_in[pkt_idx][name])

        # Verify that we selected at least as many windows as we intended to.
        num_selected_wins = len(dat_in_new)
        assert num_selected_wins >= num_wins, \
            f"Insufficient windows: {num_selected_wins} < {num_wins}"

        # As an output feature, select only the final ground truth
        # value. I.e., the final ground truth value for this window
        # becomes the ground truth for the entire window.
        return dat_in_new, np.take(dat_out, pkt_idxs), scl_grps

    def modify_data(self, sim, dat_in, dat_out, dat_extra, sequential):
        """
        Extracts from the set of simulations many separate intervals. Each
        interval becomes a training example.
        """
        dat_in, dat_out, dat_extra, scl_grps = (
            self.__create_buckets(
                sim, dat_in, dat_out, dat_extra, sequential)
            if self.rtt_buckets else (
                self.__create_windows(dat_in, dat_out, sequential)
                if self.windows else (
                    dat_in, dat_out, dat_extra,
                    # Each feature is part of its own scaling group.
                    list(range(len(dat_in.dtype.names))))))
        return dat_in, dat_out, dat_extra, scl_grps


class BinaryDnnWrapper(BinaryModelWrapper):
    """ Wraps BinaryDnn. """

    name = "BinaryDnn"
    # in_spc = ["arrival time", "loss rate"]
    in_spc = ["arrival time"]
    # in_spc = ["inter-arrival time", "loss rate"]
    # in_spc = ["inter-arrival time"]
    out_spc = ["queue occupancy"]
    los_fnc = torch.nn.CrossEntropyLoss
    opt = torch.optim.SGD
    params = ["lr", "momentum"]

    def new(self):
        self.net = BinaryDnn(self.num_ins, self.num_clss)
        return self.net


class BinaryDnn(torch.nn.Module):
    """ A simple binary classifier neural network. """

    def __init__(self, num_ins, num_outs):
        super(BinaryDnn, self).__init__()
        dim_1 = min(num_ins, 64)
        dim_2 = min(dim_1, 32)
        dim_3 = min(dim_2, 16)
        # We must store these as indivual class variables (instead of
        # just storing them in a list) because PyTorch looks at the
        # class variables to determine the model's trainable weights.
        self.fc0 = torch.nn.Linear(num_ins, dim_1)
        self.fc1 = torch.nn.Linear(dim_1, dim_2)
        self.fc2 = torch.nn.Linear(dim_2, dim_3)
        self.fc3 = torch.nn.Linear(dim_3, num_outs)
        self.sg = torch.nn.Sigmoid()
        print("BinaryDnn:\n    " +
              "\n    ".join(
                  [f"Linear: {lay.in_features}x{lay.out_features}"
                   for lay in [self.fc0, self.fc1, self.fc2, self.fc3]]) +
              "\n    Sigmoid")

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc0(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return self.sg(x)


class SvmWrapper(BinaryModelWrapper):
    """ Wraps Svm. """

    name = "Svm"
    in_spc = ["arrival time", "loss rate"]
    out_spc = ["queue occupancy"]
    los_fnc = torch.nn.HingeEmbeddingLoss
    opt = torch.optim.SGD
    params = ["lr", "momentum"]

    def new(self):
        self.net = Svm(self.num_ins)
        return self.net

    def modify_data(self, sim, dat_in, dat_out, dat_extra, sequential):
        dat_in, dat_out, dat_extra, scl_grps = (
            super(SvmWrapper, self).modify_data(
                sim, dat_in, dat_out, dat_extra, sequential))
        # Map [0,1] to [-1, 1]
        # fet = dat_out.dtype.names[0]
        # dat_out[fet] = dat_out[fet] * 2 - 1
        return dat_in, dat_out, dat_extra, scl_grps

    def _check_output_helper(self, out):
        utils.assert_tensor(out=out)
        # Remove a trailing dimension of size 1.
        out = torch.reshape(out, (out.size()[0],))
        # Transform the output to report classes -1 and 1.
        # out[torch.where(out < 0)] = -1
        # out[torch.where(out >= 0)] = 1
        return out


class Svm(torch.nn.Module):
    """ A simple SVM binary classifier. """

    def __init__(self, num_ins):
        super(Svm, self).__init__()
        self.fc = torch.nn.Linear(num_ins, 1)
        print(f"SVM:\n    Linear: {self.fc.in_features}x{self.fc.out_features}")

    def forward(self, x):
        return self.fc(x)


class SvmSklearnWrapper(SvmWrapper):
    """ Wraps an sklearn SVM. """

    name = "SvmSklearn"
    in_spc = [
        "1/sqrt loss event rate-windowed-minRtt1",
        "1/sqrt loss event rate-windowed-minRtt1024",
        "1/sqrt loss event rate-windowed-minRtt128",
        "1/sqrt loss event rate-windowed-minRtt16",
        "1/sqrt loss event rate-windowed-minRtt2",
        "1/sqrt loss event rate-windowed-minRtt256",
        "1/sqrt loss event rate-windowed-minRtt32",
        "1/sqrt loss event rate-windowed-minRtt4",
        "1/sqrt loss event rate-windowed-minRtt512",
        "1/sqrt loss event rate-windowed-minRtt64",
        "1/sqrt loss event rate-windowed-minRtt8",
        "RTT estimate ratio-ewma-alpha0.001",
        "RTT estimate ratio-ewma-alpha0.002",
        "RTT estimate ratio-ewma-alpha0.003",
        "RTT estimate ratio-ewma-alpha0.004",
        "RTT estimate ratio-ewma-alpha0.005",
        "RTT estimate ratio-ewma-alpha0.006",
        "RTT estimate ratio-ewma-alpha0.007",
        "RTT estimate ratio-ewma-alpha0.008",
        "RTT estimate ratio-ewma-alpha0.009",
        "RTT estimate ratio-ewma-alpha0.01",
        "RTT estimate ratio-ewma-alpha0.1",
        "RTT estimate ratio-ewma-alpha0.2",
        "RTT estimate ratio-ewma-alpha0.3",
        "RTT estimate ratio-ewma-alpha0.4",
        "RTT estimate ratio-ewma-alpha0.5",
        "RTT estimate ratio-ewma-alpha0.6",
        "RTT estimate ratio-ewma-alpha0.7",
        "RTT estimate ratio-ewma-alpha0.8",
        "RTT estimate ratio-ewma-alpha0.9",
        "RTT estimate ratio-ewma-alpha1.0",
        "RTT estimate us-ewma-alpha0.001",
        "RTT estimate us-ewma-alpha0.002",
        "RTT estimate us-ewma-alpha0.003",
        "RTT estimate us-ewma-alpha0.004",
        "RTT estimate us-ewma-alpha0.005",
        "RTT estimate us-ewma-alpha0.006",
        "RTT estimate us-ewma-alpha0.007",
        "RTT estimate us-ewma-alpha0.008",
        "RTT estimate us-ewma-alpha0.009",
        "RTT estimate us-ewma-alpha0.01",
        "RTT estimate us-ewma-alpha0.1",
        "RTT estimate us-ewma-alpha0.2",
        "RTT estimate us-ewma-alpha0.3",
        "RTT estimate us-ewma-alpha0.4",
        "RTT estimate us-ewma-alpha0.5",
        "RTT estimate us-ewma-alpha0.6",
        "RTT estimate us-ewma-alpha0.7",
        "RTT estimate us-ewma-alpha0.8",
        "RTT estimate us-ewma-alpha0.9",
        "RTT estimate us-ewma-alpha1.0",
        "average RTT estimate ratio-windowed-minRtt1",
        "average RTT estimate ratio-windowed-minRtt1024",
        "average RTT estimate ratio-windowed-minRtt128",
        "average RTT estimate ratio-windowed-minRtt16",
        "average RTT estimate ratio-windowed-minRtt2",
        "average RTT estimate ratio-windowed-minRtt256",
        "average RTT estimate ratio-windowed-minRtt32",
        "average RTT estimate ratio-windowed-minRtt4",
        "average RTT estimate ratio-windowed-minRtt512",
        "average RTT estimate ratio-windowed-minRtt64",
        "average RTT estimate ratio-windowed-minRtt8",
        "average RTT estimate us-windowed-minRtt1",
        "average RTT estimate us-windowed-minRtt1024",
        "average RTT estimate us-windowed-minRtt128",
        "average RTT estimate us-windowed-minRtt16",
        "average RTT estimate us-windowed-minRtt2",
        "average RTT estimate us-windowed-minRtt256",
        "average RTT estimate us-windowed-minRtt32",
        "average RTT estimate us-windowed-minRtt4",
        "average RTT estimate us-windowed-minRtt512",
        "average RTT estimate us-windowed-minRtt64",
        "average RTT estimate us-windowed-minRtt8",
        "average interarrival time us-windowed-minRtt1",
        "average interarrival time us-windowed-minRtt1024",
        "average interarrival time us-windowed-minRtt128",
        "average interarrival time us-windowed-minRtt16",
        "average interarrival time us-windowed-minRtt2",
        "average interarrival time us-windowed-minRtt256",
        "average interarrival time us-windowed-minRtt32",
        "average interarrival time us-windowed-minRtt4",
        "average interarrival time us-windowed-minRtt512",
        "average interarrival time us-windowed-minRtt64",
        "average interarrival time us-windowed-minRtt8",
        "average throughput p/s-windowed-minRtt1",
        "average throughput p/s-windowed-minRtt1024",
        "average throughput p/s-windowed-minRtt128",
        "average throughput p/s-windowed-minRtt16",
        "average throughput p/s-windowed-minRtt2",
        "average throughput p/s-windowed-minRtt256",
        "average throughput p/s-windowed-minRtt32",
        "average throughput p/s-windowed-minRtt4",
        "average throughput p/s-windowed-minRtt512",
        "average throughput p/s-windowed-minRtt64",
        "average throughput p/s-windowed-minRtt8",
        "interarrival time us-ewma-alpha0.001",
        "interarrival time us-ewma-alpha0.002",
        "interarrival time us-ewma-alpha0.003",
        "interarrival time us-ewma-alpha0.004",
        "interarrival time us-ewma-alpha0.005",
        "interarrival time us-ewma-alpha0.006",
        "interarrival time us-ewma-alpha0.007",
        "interarrival time us-ewma-alpha0.008",
        "interarrival time us-ewma-alpha0.009",
        "interarrival time us-ewma-alpha0.01",
        "interarrival time us-ewma-alpha0.1",
        "interarrival time us-ewma-alpha0.2",
        "interarrival time us-ewma-alpha0.3",
        "interarrival time us-ewma-alpha0.4",
        "interarrival time us-ewma-alpha0.5",
        "interarrival time us-ewma-alpha0.6",
        "interarrival time us-ewma-alpha0.7",
        "interarrival time us-ewma-alpha0.8",
        "interarrival time us-ewma-alpha0.9",
        "interarrival time us-ewma-alpha1.0",
        "loss event rate-windowed-minRtt1",
        "loss event rate-windowed-minRtt1024",
        "loss event rate-windowed-minRtt128",
        "loss event rate-windowed-minRtt16",
        "loss event rate-windowed-minRtt2",
        "loss event rate-windowed-minRtt256",
        "loss event rate-windowed-minRtt32",
        "loss event rate-windowed-minRtt4",
        "loss event rate-windowed-minRtt512",
        "loss event rate-windowed-minRtt64",
        "loss event rate-windowed-minRtt8",
        "loss rate estimate-ewma-alpha0.001",
        "loss rate estimate-ewma-alpha0.002",
        "loss rate estimate-ewma-alpha0.003",
        "loss rate estimate-ewma-alpha0.004",
        "loss rate estimate-ewma-alpha0.005",
        "loss rate estimate-ewma-alpha0.006",
        "loss rate estimate-ewma-alpha0.007",
        "loss rate estimate-ewma-alpha0.008",
        "loss rate estimate-ewma-alpha0.009",
        "loss rate estimate-ewma-alpha0.01",
        "loss rate estimate-ewma-alpha0.1",
        "loss rate estimate-ewma-alpha0.2",
        "loss rate estimate-ewma-alpha0.3",
        "loss rate estimate-ewma-alpha0.4",
        "loss rate estimate-ewma-alpha0.5",
        "loss rate estimate-ewma-alpha0.6",
        "loss rate estimate-ewma-alpha0.7",
        "loss rate estimate-ewma-alpha0.8",
        "loss rate estimate-ewma-alpha0.9",
        "loss rate estimate-ewma-alpha1.0",
        "loss rate estimate-windowed-minRtt1",
        "loss rate estimate-windowed-minRtt1024",
        "loss rate estimate-windowed-minRtt128",
        "loss rate estimate-windowed-minRtt16",
        "loss rate estimate-windowed-minRtt2",
        "loss rate estimate-windowed-minRtt256",
        "loss rate estimate-windowed-minRtt32",
        "loss rate estimate-windowed-minRtt4",
        "loss rate estimate-windowed-minRtt512",
        "loss rate estimate-windowed-minRtt64",
        "loss rate estimate-windowed-minRtt8",
        "mathis model label-ewma-alpha0.001",
        "mathis model label-ewma-alpha0.002",
        "mathis model label-ewma-alpha0.003",
        "mathis model label-ewma-alpha0.004",
        "mathis model label-ewma-alpha0.005",
        "mathis model label-ewma-alpha0.006",
        "mathis model label-ewma-alpha0.007",
        "mathis model label-ewma-alpha0.008",
        "mathis model label-ewma-alpha0.009",
        "mathis model label-ewma-alpha0.01",
        "mathis model label-ewma-alpha0.1",
        "mathis model label-ewma-alpha0.2",
        "mathis model label-ewma-alpha0.3",
        "mathis model label-ewma-alpha0.4",
        "mathis model label-ewma-alpha0.5",
        "mathis model label-ewma-alpha0.6",
        "mathis model label-ewma-alpha0.7",
        "mathis model label-ewma-alpha0.8",
        "mathis model label-ewma-alpha0.9",
        "mathis model label-ewma-alpha1.0",
        "mathis model label-windowed-minRtt1",
        "mathis model label-windowed-minRtt1024",
        "mathis model label-windowed-minRtt128",
        "mathis model label-windowed-minRtt16",
        "mathis model label-windowed-minRtt2",
        "mathis model label-windowed-minRtt256",
        "mathis model label-windowed-minRtt32",
        "mathis model label-windowed-minRtt4",
        "mathis model label-windowed-minRtt512",
        "mathis model label-windowed-minRtt64",
        "mathis model label-windowed-minRtt8",
        "mathis model throughput p/s-ewma-alpha0.001",
        "mathis model throughput p/s-ewma-alpha0.002",
        "mathis model throughput p/s-ewma-alpha0.003",
        "mathis model throughput p/s-ewma-alpha0.004",
        "mathis model throughput p/s-ewma-alpha0.005",
        "mathis model throughput p/s-ewma-alpha0.006",
        "mathis model throughput p/s-ewma-alpha0.007",
        "mathis model throughput p/s-ewma-alpha0.008",
        "mathis model throughput p/s-ewma-alpha0.009",
        "mathis model throughput p/s-ewma-alpha0.01",
        "mathis model throughput p/s-ewma-alpha0.1",
        "mathis model throughput p/s-ewma-alpha0.2",
        "mathis model throughput p/s-ewma-alpha0.3",
        "mathis model throughput p/s-ewma-alpha0.4",
        "mathis model throughput p/s-ewma-alpha0.5",
        "mathis model throughput p/s-ewma-alpha0.6",
        "mathis model throughput p/s-ewma-alpha0.7",
        "mathis model throughput p/s-ewma-alpha0.8",
        "mathis model throughput p/s-ewma-alpha0.9",
        "mathis model throughput p/s-ewma-alpha1.0",
        "mathis model throughput p/s-windowed-minRtt1",
        "mathis model throughput p/s-windowed-minRtt1024",
        "mathis model throughput p/s-windowed-minRtt128",
        "mathis model throughput p/s-windowed-minRtt16",
        "mathis model throughput p/s-windowed-minRtt2",
        "mathis model throughput p/s-windowed-minRtt256",
        "mathis model throughput p/s-windowed-minRtt32",
        "mathis model throughput p/s-windowed-minRtt4",
        "mathis model throughput p/s-windowed-minRtt512",
        "mathis model throughput p/s-windowed-minRtt64",
        "mathis model throughput p/s-windowed-minRtt8",
        "min RTT us",
        "throughput p/s-ewma-alpha0.001",
        "throughput p/s-ewma-alpha0.002",
        "throughput p/s-ewma-alpha0.003",
        "throughput p/s-ewma-alpha0.004",
        "throughput p/s-ewma-alpha0.005",
        "throughput p/s-ewma-alpha0.006",
        "throughput p/s-ewma-alpha0.007",
        "throughput p/s-ewma-alpha0.008",
        "throughput p/s-ewma-alpha0.009",
        "throughput p/s-ewma-alpha0.01",
        "throughput p/s-ewma-alpha0.1",
        "throughput p/s-ewma-alpha0.2",
        "throughput p/s-ewma-alpha0.3",
        "throughput p/s-ewma-alpha0.4",
        "throughput p/s-ewma-alpha0.5",
        "throughput p/s-ewma-alpha0.6",
        "throughput p/s-ewma-alpha0.7",
        "throughput p/s-ewma-alpha0.8",
        "throughput p/s-ewma-alpha0.9",
        "throughput p/s-ewma-alpha1.0"
    ]
    in_spc = [fet for fet in in_spc if (
        "alpha0.002" not in fet and
        "alpha0.004" not in fet and
        "alpha0.006" not in fet and
        "alpha0.008" not in fet and
        "alpha0.01" not in fet and
        "alpha0.2" not in fet and
        "alpha0.4" not in fet and
        "alpha0.6" not in fet and
        "alpha0.7" not in fet and
        "alpha0.8" not in fet and
        "alpha0.9" not in fet and
        "tt512" not in fet and
        "tt1024" not in fet and
        "mathis model" not in fet)]
    out_spc = ["queue occupancy-ewma-alpha0.01"]
    los_fnc = None
    opt = None
    params = ["kernel", "degree", "penalty", "max_iter", "graph"]
    num_clss = 2
    multiclass = num_clss > 2

    def new(self, **kwargs):
        self.graph = kwargs["graph"]
        kernel = kwargs["kernel"]
        max_iter = kwargs["max_iter"]
        assert not self.multiclass or kernel == "linear", \
            "Kernel must be linear for multiclass mode."
        # Automatically set the class weights based on the class
        # popularity in the training data. Change the maximum number
        # of iterations.
        self.net = (
            # Use manually-configured regularization. Since the number
            # of samples is greater than the number of features, solve
            # the primal optimization problem instead of its dual.
            svm.LinearSVC(
                penalty=kwargs["penalty"], dual=False, class_weight="balanced",
                verbose=1, max_iter=max_iter, multi_class="ovr")
            if kernel == "linear" else
            # Supports L2 regularization only. The degree parameter is
            # used only if kernel == "poly".
            svm.SVC(
                kernel=kernel, degree=kwargs["degree"], class_weight="balanced",
                verbose=1, max_iter=max_iter))
        return self.net

    def train(self, dat_in, dat_out):
        """ Fits this model to the provided dataset. """
        self.net.fit(dat_in, dat_out)
        # Oftentimes, the training log statements do not end with a newline.
        print()

    @staticmethod
    def convert_to_class(sim, dat_out):
        if SvmSklearnWrapper.num_clss == 2:
            return BinaryModelWrapper.convert_to_class(sim, dat_out)
        assert SvmSklearnWrapper.num_clss == 3, \
            ("Only 2 or 3 classes are supported, not: "
             f"{SvmSklearnWrapper.num_clss}")
        assert len(dat_out.dtype.names) == 1, "Should be only one column."

        def percent_to_class(prc, fair):
            """ Convert a queue occupancy percent to a fairness class. """
            assert len(prc) == 1, "Should be only one column."
            prc = prc[0]

            # Threshold between fair and unfair.
            tsh_fair = 0.1

            dif = (fair - prc) / fair
            if dif < -1 * tsh_fair:
                # We are much higher than fair.
                cls = 2
            elif -1 * tsh_fair <= dif <= tsh_fair:
                # We are fair.
                cls = 1
            elif tsh_fair < dif:
                # We are much lower than fair.
                cls = 0
            else:
                assert False, "This should never happen."
            return cls

        # Map a conversion function across all entries. Note that here
        # an entry is an entire row, since each row is a single tuple
        # value.
        clss = np.vectorize(
            functools.partial(
                percent_to_class, fair=1. / (sim.unfair_flws + sim.fair_flws)),
            otypes=[int])(dat_out)
        clss_str = np.empty((clss.shape[0],), dtype=[("class", "int")])
        clss_str["class"] = clss
        return clss_str


    def __evaluate(self, preds, labels, raw, fair, sort_by_unfairness=False,
                   graph_prms=None):
        """
        Returns the accuracy of predictions compared to ground truth
        labels. If self.graph == True, then this function also graphs
        the accuracy. preds, labels, raw, and fair must be Torch tensors.
        """
        utils.assert_tensor(preds=preds, labels=labels, raw=raw, fair=fair)
        # Compute the distance from fair, then divide by fair to
        # compute the relative unfairness.
        diffs = (raw - fair) / fair
        if sort_by_unfairness:
            # Sort based on unfairness.
            diffs, indices = torch.sort(diffs)
            preds = preds[indices]
            labels = labels[indices]
        # Bucketize and compute bucket accuracies.
        num_samples = preds.size()[0]
        num_buckets = min(20 * (1 if sort_by_unfairness else 4), num_samples)
        num_per_bucket = math.floor(num_samples / num_buckets)
        assert num_per_bucket > 0, \
            ("There must be at least one sample per bucket, but there are "
             f"{num_samples} samples and only {num_buckets} buckets!")
        # The resulting buckets are tuples of three values:
        #   (x-axis value for bucket, number predicted correctly, total)
        buckets = [
            (x,
             self.check_output(preds_, labels_),
             preds_.size()[0])
            for x, preds_, labels_ in [
                # Each bucket is defined by a tuple of three values:
                #   (x-axis value for bucket, predictions, ground truth labels).
                # The x-axis is the mean relative difference for this
                # bucket. A few values at the end may be discarded.
                (torch.mean(diffs[i:i + num_per_bucket]),
                 preds[i:i + num_per_bucket],
                 labels[i:i + num_per_bucket])
                for i in range(0, num_samples, num_per_bucket)]]
        if self.graph:
            assert graph_prms is not None, \
                "\"graph_prms\" must be a dict(), not None."
            assert "flp" in graph_prms, "\"flp\" not in \"graph_prms\"!"
            assert "x_lim" in graph_prms, "\"x_lim\" not in \"graph_prms\"!"
            # Plot each bucket's accuracy.
            pyplot.plot(
                ([x for x, _, _ in buckets]
                 if sort_by_unfairness else list(range(len(buckets)))),
                [c / t for _, c, t in buckets], "bo-")
            pyplot.ylim((-0.1, 1.1))
            x_lim = graph_prms["x_lim"]
            if x_lim is not None:
                pyplot.xlim(x_lim)
            pyplot.xlabel(
                "Unfairness (fraction of fair)"
                if sort_by_unfairness else "Time")
            pyplot.ylabel("Classification accuracy")
            pyplot.tight_layout()
            pyplot.savefig(graph_prms["flp"])
            pyplot.close()
        # Compute the overall accuracy.
        _, corrects, totals = zip(*buckets)
        acc = sum(corrects) / sum(totals)
        print(f"    Test accuracy: {acc * 100:.2f}%")
        return acc


    def __evaluate_sliding_window(self, preds, raw, fair, arr_times,
                                  rtt_estimates_us):
        """
        Returns the sliding window accuracy of predictions based
        on the rtt estimate and the window size. all arguments must be Torch
        tensors.

        preds: Prediction produced by the model
        raw: Raw values of the queue occupancy
        fair: Fair share of the flow
        arr_times: The arrival times of each sample.
        rtt_estimates_us: Rtt estimates computed on the receiver side
        """
        utils.assert_tensor(
            preds=preds, raw=raw, fair=fair, arr_times=arr_times,
            rtt_estimates_us=rtt_estimates_us)
        num_pkts = len(raw)
        assert len(preds) == num_pkts
        assert len(fair) == num_pkts
        assert len(arr_times) == num_pkts
        assert len(rtt_estimates_us) == num_pkts

        # Compute sliding window accuracy based on arrival time and RTT
        sliding_window_accuracy = [0] * num_pkts
        window_head = 0
        max_packet = 0
        for i in range(num_pkts):
            recv_time = arr_times[i]
            window_size_us = SLIDING_WINDOW_NUM_RTT * rtt_estimates_us[i]
            while (window_head < num_pkts and
                   recv_time - arr_times[window_head] >= window_size_us):
                window_head += 1

            max_packet = max(max_packet, i - window_head)
            queue_occupancy = torch.mean(raw[window_head:i + 1])
            label = torch.mean(preds[window_head:i + 1].type(torch.float))
            sliding_window_accuracy[i] = (
                int(label >= SMOOTHING_THRESHOLD)
                if queue_occupancy > fair[i] else
                int(label < SMOOTHING_THRESHOLD))
        return sum(sliding_window_accuracy) / len(sliding_window_accuracy)

    def __plot_throughput(self, labels, preds, fair, flp, btk_throughput,
                          throughput_ewma, x_lim=None):
        """
        Plots the fair flow's throughput over time.

        labels: Ground truth labels.
        preds: Prediction produced by the model
        fair: Fair share of the flow
        flp: File name of the saved graph.
        btk_throughput: Throughput of the bottleneck link
        throughput_ewma: Throughput ewma computed in parse_dumbbell
        x_lim: Graph limit on the x axis."""

        fair = fair * btk_throughput
        throughput_ewma = throughput_ewma * 1380 * 8 / 1000000

        if self.graph:
            # Bucketize and compute bucket accuracies.
            num_samples = len(labels)
            num_buckets = min(20 * 4, num_samples)
            num_per_bucket = math.floor(num_samples / num_buckets)

            assert num_per_bucket > 0, \
            ("There must be at least one sample per bucket, but there are "
             f"{num_samples} samples and only {num_buckets} buckets!")

            buckets = [
                (torch.mean(throughput_),
                 torch.mean(fair_),
                 self.check_output(preds_, labels_) / num_per_bucket,
                 )
                for throughput_, fair_, preds_, labels_, in [
                    (throughput_ewma[i:i + num_per_bucket],
                     fair[i:i + num_per_bucket],
                     preds[i:i + num_per_bucket],
                     labels[i:i + num_per_bucket]
                     )
                    for i in range(0, num_samples, num_per_bucket)]]

            throughput_list, fair_list, accuracy_list = zip(*buckets)

            x_axis = list(range(len(buckets)))

            fig, ax1 = pyplot.subplots()
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Throughput")
            ax1.plot(x_axis, throughput_list, "b-")
            ax1.plot(x_axis, fair_list, "g--")

            ax2 = ax1.twinx()

            ax2.set_ylabel("Accuracy")
            ax2.plot(x_axis, accuracy_list, "r-")

            if x_lim is not None:
                pyplot.xlim(x_lim)
            fig.tight_layout()
            pyplot.savefig(flp)
            pyplot.close()


    def __plot_queue_occ(self, raw, fair, flp, x_lim=None):
        """
        Plots the queue occupancy over time. labels, raw, and fair must be
        Torch tensors.

        raw: Raw values of the queue occupancy
        fair: Fair share of the flow
        flp: File name of the saved graph.
        x_lim: Graph limit on the x axis."""

        if self.graph:
            # Bucketize and compute bucket accuracies.
            num_samples = len(raw.tolist())
            num_buckets = min(20 * 4, num_samples)
            num_per_bucket = math.floor(num_samples / num_buckets)

            assert num_per_bucket > 0, \
            ("There must be at least one sample per bucket, but there are "
             f"{num_samples} samples and only {num_buckets} buckets!")

            buckets = [
                (torch.mean(raw[i:i + num_per_bucket]),
                 torch.mean(fair[i:i + num_per_bucket]))
                for i in range(0, num_samples, num_per_bucket)]
            queue_occupancy_list, fair_list = zip(*buckets)
            x_axis = list(range(len(buckets)))
            pyplot.plot(x_axis, queue_occupancy_list, "b-")
            pyplot.plot(x_axis, fair_list, "g--")

            if x_lim is not None:
                pyplot.xlim(x_lim)
            pyplot.xlabel("Time")
            pyplot.ylabel("Queue occupancy vs Fair queue occupancy")
            pyplot.tight_layout()
            pyplot.savefig(flp)
            pyplot.close()


    def test(self, fets, dat_in, dat_out_classes, dat_extra,
             graph_prms=copy.copy({
                 "out_dir": ".", "sort_by_unfairness": True, "dur_s": None})):
        """
        Tests this model on the provided dataset and returns the test accuracy
        (higher is better). Also, analyzes the model's feature coefficients and
        (if self.graph == True) visualizes various metrics. dat_in and
        dat_out_classes must be Torch tensors. dat_extra must be a Numpy array.

        fets: List of feature names.
        dat_in: Test data.
        dat_out_classes: Ground truth labels.
        dat_extra: Extra data for each sample.
        graph_prms: Graphing parameters. Used only if self.graph == True.
        """
        utils.assert_tensor(dat_in=dat_in, dat_out_classes=dat_out_classes)
        sort_by_unfairness = graph_prms["sort_by_unfairness"]
        dur_s = graph_prms["dur_s"]
        assert sort_by_unfairness or dur_s is not None, \
            ("If \"sort_by_unfairness\" is False, then \"dur_s\" must not be "
             "None.")

        # Compute the fair share. Convert from int to float to avoid
        # all values being rounded to 0.
        fair = np.reciprocal(dat_extra["num_flws"].astype(float))
        # Calculate the x limits. Determine the maximum unfairness.
        x_lim = (
            # Compute the maximum unfairness.
            (-1, ((dat_extra["raw"] - fair) / fair).max().item())
            if sort_by_unfairness else (0, graph_prms["dur_s"]))
        # Optionally create the output directory.
        out_dir = path.join(graph_prms["out_dir"], self.name)
        if self.graph:
            # Create the output directory.
            if not path.exists(out_dir):
                os.makedirs(out_dir)

        # Analyze feature coefficients. The underlying model's .coef_
        # attribute may not exist.
        try:
            if isinstance(
                    self.net,
                    (sklearn.feature_selection.RFE,
                     sklearn.feature_selection.RFECV)):
                # Since the model was trained using RFE, display all
                # features. Sort the features alphabetically.
                best_fets = sorted(
                    zip(
                        np.array(fets)[np.where(self.net.ranking_ == 1)],
                        self.net.estimator_.coef_[0]),
                    key=lambda p: p[0])
                print(f"Number of features selected: {len(best_fets)}")
                qualifier = "All"
            else:
                # First, sort the features by the absolute value of the
                # importance and pick the top 20. Then, sort the features
                # alphabetically.
                best_fets = sorted(
                    sorted(
                        zip(fets, self.net.coef_[0]),
                        key=lambda p: abs(p[1]))[-20:],
                    key=lambda p: p[0])
                qualifier = "Best"
            print(
                f"----------\n{qualifier} features ({len(best_fets)}):\n" +
                "\n".join([f"{fet}: {coef}" for fet, coef in best_fets]) +
                "\n----------")
            if self.graph:
                names, coefs = zip(*best_fets)
                num_fets = len(names)
                y_vals = list(range(num_fets))
                pyplot.figure(figsize=(7, 0.2 * num_fets))
                pyplot.barh(y_vals, coefs, align="center")
                pyplot.yticks(y_vals, names)
                pyplot.ylim((-1, num_fets))
                pyplot.xlabel("Feature coefficient")
                pyplot.ylabel("Feature name")
                pyplot.tight_layout()
                pyplot.savefig(path.join(out_dir, f"features_{self.name}.pdf"))
                pyplot.close()
        except AttributeError:
            # Coefficients are only available with a linear kernel.
            print("Warning: Unable to extract coefficients!")

        if self.graph:
            # Analyze, for each number of flows, accuracy vs. unfairness.
            flws_accs = []
            nums_flws = list(set(dat_extra["num_flws"].tolist()))
            for num_flws_selected in nums_flws:
                print(f"Evaluating model for {num_flws_selected} flows:")
                valid = np.where(dat_extra["num_flws"] == num_flws_selected)
                flws_accs.append(self.__evaluate(
                    torch.tensor(self.net.predict(dat_in[valid])),
                    dat_out_classes[valid],
                    torch.tensor(dat_extra["raw"][valid]),
                    torch.tensor(fair[valid]),
                    sort_by_unfairness,
                    graph_prms={
                        "flp": path.join(
                            out_dir,
                            (f"accuracy_vs_unfairness_{num_flws_selected}flows_"
                             f"{self.name}.pdf")),
                        "x_lim": x_lim}))

            # Analyze accuracy vs. number of flows.
            x_vals = list(range(len(flws_accs)))
            pyplot.bar(x_vals, flws_accs, align="center")
            pyplot.xticks(x_vals, nums_flws)
            pyplot.ylim((0, 1.1))
            pyplot.xlabel("Total flows (1 unfair)")
            pyplot.ylabel("Classification accuracy")
            pyplot.tight_layout()
            pyplot.savefig(
                path.join(out_dir, f"accuracy_vs_num-flows_{self.name}.pdf"))
            pyplot.close()

            # Plot queue occupancy.
            self.__plot_queue_occ(
                torch.tensor(dat_extra["raw"]),
                torch.tensor(fair),
                path.join(
                    out_dir, f"queue_occ_vs_fair_queue_occ_{self.name}.pdf"),
                x_lim)

            # Plot throughput
            self.__plot_throughput(dat_out_classes,
                    torch.tensor(self.net.predict(dat_in)),
                    torch.tensor(fair), path.join(
                    out_dir, f"throughput_{self.name}.pdf"),
                    dat_extra["btk_throughput"],
                    torch.tensor(dat_extra[defaults.THR_ESTIMATE_FET].copy()),
                    x_lim=None)

        # Analyze accuracy vs. unfairness for all flows and all
        # degrees of unfairness, for the Mathis Model.
        print("Evaluting Mathis Model:")
        raw = dat_extra["raw"].copy()
        self.__evaluate(
            torch.tensor(dat_extra[defaults.MATHIS_MODEL_FET].copy()),
            dat_out_classes, torch.tensor(raw), torch.tensor(fair),
            sort_by_unfairness,
            graph_prms={
                "flp": path.join(
                    out_dir, "accuracy_vs_unfairness_mathis.pdf"),
                "x_lim": x_lim})

        predictions = torch.tensor(self.net.predict(dat_in))

        # Analyze accuracy vs unfairness for all flows and all degrees
        # of unfairness, for the model itself.
        print(f"Evaluating {self.name} model:")
        model_acc = self.__evaluate(
            predictions, dat_out_classes, torch.tensor(raw), torch.tensor(fair),
            sort_by_unfairness,
            graph_prms={
                "flp": path.join(
                    out_dir, f"accuracy_vs_unfairness_{self.name}.pdf"),
                "x_lim": x_lim})

        # Analyze accuracy of a sliding window method
        sliding_window_accuracy = self.__evaluate_sliding_window(
            predictions, torch.tensor(raw), torch.tensor(fair),
            torch.tensor(dat_extra[defaults.ARRIVAL_TIME_FET].copy()),
            torch.tensor(dat_extra[defaults.RTT_ESTIMATE_FET].copy()))

        return model_acc, sliding_window_accuracy


class LrSklearnWrapper(SvmSklearnWrapper):
    """ Wraps an sklearn Logistic Regression model. """

    name = "LrSklearn"
    params = ["max_iter", "rfe", "graph"]

    @staticmethod
    def rfe(net, rfe_type):
        """ Apply recursive feature elimination to the provided net. """
        final_net = net
        if rfe_type == "None":
            print("Not using recursive feature elimination.")
        elif rfe_type == "rfe":
            print("Using recursive feature elimination.")
            final_net = sklearn.feature_selection.RFE(
                estimator=net, n_features_to_select=10, step=10)
        elif rfe_type == "rfecv":
            print("Using recursive feature elimination with cross-validation.")
            final_net = sklearn.feature_selection.RFECV(
                estimator=net, step=1,
                cv=sklearn.model_selection.StratifiedKFold(10),
                scoring="accuracy", n_jobs=-1)
        else:
            raise Exception(f"Unknown RFE type: {rfe_type}")
        return final_net

    def new(self, **kwargs):
        self.graph = kwargs["graph"]
        # Use L1 regularization. Since the number of samples is
        # greater than the number of features, solve the primal
        # optimization problem instead of its dual. Automatically set
        # the class weights based on the class popularity in the
        # training data. Change the maximum number of iterations.
        self.net = self.rfe(
            linear_model.LogisticRegression(
                penalty="l1", dual=False, class_weight="balanced",
                solver="saga", max_iter=kwargs["max_iter"], verbose=1,
                multi_class="ovr", n_jobs=-1),
            kwargs["rfe"])
        return self.net


class LrCvSklearnWrapper(LrSklearnWrapper):
    """
    Wraps an sklearn Logistic Regression model, but uses cross-validation.
    """

    name = "LrCvSklearn"
    params = ["max_iter", "rfe", "graph", "folds"]

    def new(self, **kwargs):
        # Use L1 regularization. Since the number of samples is
        # greater than the number of features, solve the primal
        # optimization problem instead of its dual. Automatically set
        # the class weights based on the class popularity in the
        # training data. Change the maximum number of iterations.
        # Use the specified number of cross-validation folds. Use
        # all cores.
        self.net = self.rfe(
            linear_model.LogisticRegressionCV(
                cv=kwargs["folds"], penalty="l1", dual=False,
                class_weight="balanced", solver="saga",
                max_iter=kwargs["max_iter"], verbose=1, n_jobs=-1,
                multi_class="ovr"),
            kwargs["rfe"])
        return self.net


class LstmWrapper(PytorchModelWrapper):
    """ Wraps Lstm. """

    name = "Lstm"
    # For now, we do not use RTT ratio because our method of estimating
    # it cannot be performed by a general receiver.
    # in_spc = ["inter-arrival time", "RTT ratio", "loss rate"]
    in_spc = ["inter-arrival time", "loss rate"]
    out_spc = ["queue occupancy"]
    num_clss = 5
    # Cross-entropy loss is designed for multi-class classification tasks.
    los_fnc = torch.nn.CrossEntropyLoss
    opt = torch.optim.Adam
    params = ["lr"]

    def __init__(self, hid_dim=32, num_lyrs=1, out_dim=5):
        super(LstmWrapper, self).__init__()
        self.in_dim = len(self.in_spc)
        self.hid_dim = hid_dim
        self.num_lyrs = num_lyrs
        self.out_dim = out_dim

    def new(self):
        self.net = Lstm(self.in_dim, self.hid_dim, self.num_lyrs, self.out_dim)
        return self.net

    def init_hidden(self, batch_size):
        weight = next(self.net.parameters()).data
        return (weight.new(self.num_lyrs, batch_size, self.hid_dim).zero_(),
                weight.new(self.num_lyrs, batch_size, self.hid_dim).zero_())

    @staticmethod
    def convert_to_class(sim, dat_out):
        assert len(dat_out.dtype.names) == 1, "Should be only one column."

        def percent_to_class(prc, fair):
            """ Convert a queue occupancy percent to a fairness class. """
            assert len(prc) == 1, "Should be only one column."
            prc = prc[0]

            # Threshold between fair and unfair.
            tsh_fair = 0.1
            # Threshold between unfair and very unfair.
            tsh_unfair = 0.4

            dif = (fair - prc) / fair
            if dif < -1 * tsh_unfair:
                # We are much higher than fair.
                cls = 4
            elif -1 * tsh_unfair <= dif < -1 * tsh_fair:
                # We are not that much higher than fair.
                cls = 3
            elif -1 * tsh_fair <= dif <= tsh_fair:
                # We are fair.
                cls = 2
            elif tsh_fair < dif <= tsh_unfair:
                # We are not that much lower than fair.
                cls = 1
            elif tsh_unfair < dif:
                # We are much lower than fair.
                cls = 0
            else:
                assert False, "This should never happen."
            return cls

        # Map a conversion function across all entries. Note that here
        # an entry is an entire row, since each row is a single tuple
        # value.
        clss = np.vectorize(
            functools.partial(
                percent_to_class, fair=1. / (sim.unfair_flws + sim.fair_flws)),
            otypes=[int])(dat_out)
        clss_str = np.empty((clss.shape[0],), dtype=[("class", "int")])
        clss_str["class"] = clss
        return clss_str


class Lstm(torch.nn.Module):
    """ An LSTM that classifies a flow into one of five fairness categories. """

    def __init__(self, in_dim, hid_dim, num_lyrs, out_dim):
        super(Lstm, self).__init__()
        self.hid_dim = hid_dim
        self.lstm = torch.nn.LSTM(in_dim, self.hid_dim)
        self.fc = torch.nn.Linear(self.hid_dim, out_dim)
        self.sg = torch.nn.Sigmoid()
        print(f"Lstm - in_dim: {in_dim}, hid_dim: {self.hid_dim}, "
              f"num_lyrs: {num_lyrs}, out_dim: {out_dim}")

    def forward(self, x, hidden):
        # The LSTM itself, which also takes as input the previous hidden state.
        out, hidden = self.lstm(x, hidden)
        # Select the last piece as the LSTM output.
        out = out.contiguous().view(-1, self.hid_dim)
        # Classify the LSTM output.
        out = self.fc(out)
        out = self.sg(out)
        return out, hidden


#######################################################################
#### Old models. Present for archival purposes only. These are not ####
#### guaranteed to function.                                       ####
#######################################################################


class SimpleOne(torch.nn.Module):
    """ A simple linear model. """

    in_spc = ["average_throughput_after_bps"]
    out_spc = ["ack_period_us"]

    def __init__(self):
        super(SimpleOne, self).__init__()
        assert len(SimpleOne.in_spc) == 1
        assert len(SimpleOne.out_spc) == 1
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


class FcOne(torch.nn.Module):
    """ An NN with one fully-connected layer. """

    in_spc = ["average_throughput_before_bps", "average_throughput_after_bps",
              "rtt_us"]
    out_spc = SimpleOne.out_spc

    def __init__(self):
        super(FcOne, self).__init__()
        assert len(FcOne.in_spc) == 3
        assert len(FcOne.out_spc) == 1
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


class FcTwo(torch.nn.Module):
    """ An NN with two fully-connected layers. """

    in_spc = FcOne.in_spc
    out_spc = SimpleOne.out_spc

    def __init__(self):
        super(FcTwo, self).__init__()
        assert len(FcTwo.in_spc) == 3
        assert len(FcTwo.out_spc) == 1
        self.fc1 = torch.nn.Linear(3, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return x


class FcThree(torch.nn.Module):
    """ An NN with three fully-connected layers. """

    in_spc = FcOne.in_spc
    out_spc = SimpleOne.out_spc

    def __init__(self):
        super(FcThree, self).__init__()
        assert len(FcThree.in_spc) == 3
        assert len(FcThree.out_spc) == 1
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return x


class FcFour(torch.nn.Module):
    """ An NN with four fully-connected layers. """

    in_spc = FcOne.in_spc
    out_spc = SimpleOne.out_spc

    def __init__(self):
        super(FcFour, self).__init__()
        assert len(FcFour.in_spc) == 3
        assert len(FcFour.out_spc) == 1
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        return x


#######################################################################


MODELS = {mdl.name: mdl for mdl in [
    BinaryDnnWrapper,
    SvmWrapper,
    SvmSklearnWrapper,
    LrSklearnWrapper,
    LrCvSklearnWrapper,
    LstmWrapper
]}
MODEL_NAMES = sorted(MODELS.keys())
