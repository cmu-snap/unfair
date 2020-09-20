""" Default values. """

import sys

import models


# Parameter defaults.
DEFAULTS = {
    "data_dir": ".",
    "warmup_percent": 0,
    "keep_percent": 100,
    "num_sims": sys.maxsize,
    "sims": [],
    "model": models.MODEL_NAMES[0],
    "features": [],
    "epochs": 100,
    "num_gpus": 0,
    "train_batch": sys.maxsize,
    "test_batch": sys.maxsize,
    "learning_rate": 0.001,
    "momentum": 0.09,
    "kernel": "linear",
    "degree": 3,
    "penalty": "l1",
    "max_iter": 10000,
    "rfe": "None",
    "folds": 2,
    "graph": False,
    "standardize": True,
    "early_stop": False,
    "val_patience": 10,
    "val_improvement_thresh": 0.1,
    "conf_trials": 1,
    "max_attempts": 10,
    "no_rand": False,
    "timeout_s": 0,
    "out_dir": ".",
    "tmp_dir": None,
    "regen_data": False,
    "sync": False
}
# The maximum number of epochs when using early stopping.
EPCS_MAX = 10_000
# Whether to execute synchronously or in parallel.
SYNC = False
# Features to store as extra data for each sample.
MATHIS_MODEL_FET = "mathis model label-ewma-alpha0.01"
RTT_ESTIMATE_FET = "RTT estimate us-ewma-alpha0.01"
ARRIVAL_TIME_FET = "arrival time us"
THR_ESTIMATE_FET = "throughput p/s-ewma-alpha0.007"
EXTRA_FETS = [ARRIVAL_TIME_FET, MATHIS_MODEL_FET,
              RTT_ESTIMATE_FET, THR_ESTIMATE_FET]
