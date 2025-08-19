import os
import sys
import json
import argparse
import warnings
import pickle

sys.path.append("..")

import time
from datetime import timedelta
import random
import pandas as pd
import wandb

import numpy as np
from sklearn.utils import class_weight
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

from simuaug.utils import get_default_device, paint, seed_torch
from simuaug.datasets import SensorDataset
from simuaug.augmentations import AugPolicyPlanner
from simuaug.augmentations.ppda import (
    Identity,
    MagnitudeScaling,
    NoiseBias,
    TimeWarping,
    TimeScaling,
    Rotation,
    MagnitudeWarping,
)

from models import DeepConvLSTM, AttendAndDiscriminate
from models.utils import init_weights, wandb_logging
from models.utils import run_train_analysis, run_test_analysis

from scripts.utils import eval_model, eval_one_epoch, train_one_epoch
from simuaug.dataloaders import PPDADataLoader
from wimusim.datasets import WIMUSimDataset


def train_model_with_ppda(
    model,
    train_data,
    train_data_sim,
    val_data,
    test_data,
    batch_size_train,
    aug_planner,
    seed=1,
    class_weights=None,
    verbose=True,
    n_epochs=100,
    lr=1e-3,
    lr_a=5 * 1e-3,  # learning rate for the augmentation parameters
    lr_step=10,
    lr_decay=0.9,
    weights_init="orthogonal",
    batch_size_test=256,
    aug_first=True,
):
    if verbose:
        print(
            paint(
                f"================= Running HAR training loop with seed {seed} ================="
            )
        )

    # Loader for original dataset
    loader_train = DataLoader(
        train_data,
        batch_size=batch_size_train,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
    )

    # Loader for the augmented dataset
    loader_train_aug = PPDADataLoader(
        train_data_sim,
        batch_size=batch_size_train,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
        aug_planner=aug_planner,
        sample=True,
    )

    loader_val = DataLoader(
        val_data,
        batch_size=batch_size_train,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
    )

    loader_test = DataLoader(
        test_data,
        batch_size=batch_size_test,
        shuffle=False,
        worker_init_fn=np.random.seed(int(seed)),
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_decay
    )

    init_weights(model, weights_init)

    metric_best = 0.0

    # Store the training and validation metrics
    t_loss, t_acc, t_fm, t_fw = [], [], [], []
    v_loss, v_acc, v_fm, v_fw = [], [], [], []

    for epoch in range(n_epochs):
        if verbose:
            print(
                f"------------------------------ Epoch {epoch + 1}/{n_epochs} ------------------------------\n"
                f"Learning rate: {optimizer.param_groups[0]['lr']}, tau: {aug_planner.tau}"
            )
        aug_planner.update_tau(epoch)

        # 50% Identity, 50% Augmentation
        train_one_epoch(model, loader_train_aug, criterion, optimizer)

        loss, acc, fm, fw = eval_one_epoch(model, loader_train, criterion)
        loss_val, acc_val, fm_val, fw_val = eval_one_epoch(model, loader_val, criterion)

        # Check performance on the test dataset
        loss_test, acc_test, fm_test, fw_test = eval_one_epoch(
            model, loader_test, criterion
        )

        # Store the metrics
        t_loss.append(loss)
        t_acc.append(acc)
        t_fm.append(fm)
        t_fw.append(fw)
        v_loss.append(loss_val)
        v_acc.append(acc_val)
        v_fm.append(fm_val)
        v_fw.append(fw_val)

        if verbose:
            print(
                paint(
                    f"\tTrain loss: {loss:.2f} \tacc: {100 * acc:.2f}(%)\tfm: {100 * fm:.2f}(%)\tfw: {100 * fw:.2f}"
                    f"(%)\t"
                )
            )

            print(
                paint(
                    f"\tVal loss:   {loss_val:.2f} \tacc: {100 * acc_val:.2f}(%)\tfm: {100 * fm_val:.2f}(%)"
                    f"\tfw: {100 * fw_val:.2f}(%)"
                )
            )

            print(
                paint(
                    f"\tTest loss:  {loss_test:.2f} \tacc: {100 * acc_test:.2f}(%)\tfm: {100 * fm_test:.2f}(%)"
                    f"\tfw: {100 * fw_test:.2f}(%)"
                )
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:  # Ignore first 5 epochs
            if verbose:
                print(
                    paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue")
                )
            # Don't update the best metric for the first 5 epochs as it's unstable
            metric_best = metric
            torch.save(checkpoint, os.path.join(model.path_checkpoints, "best.pth"))

        if lr_step > 0:
            scheduler.step()

    return t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, criterion


# Define a helper function to parse the tuple input
def parse_range(range_str):
    try:
        # Split the input by comma and convert to a tuple of floats
        min_val, max_val = map(float, range_str.split(","))
        return (min_val, max_val)  # Return as a tuple
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Rotation range must be in the form 'min,max'."
        )


def get_config():
    parser = argparse.ArgumentParser(description="Configuration for experiment")

    # Adding all the parameters from your config
    parser.add_argument(
        "--model", type=str, default="DeepConvLSTM", help="DeepLearning model"
    )
    parser.add_argument("--window", type=int, default=100, help="Window size")
    parser.add_argument("--stride", type=int, default=25, help="Stride size")
    parser.add_argument(
        "--dataset_name", type=str, default="realdisp", help="Name of the dataset"
    )
    parser.add_argument(
        "--train_prefix",
        type=str,
        default="train-phase2",
        help="Prefix for training data",
    )
    parser.add_argument(
        "--val_prefix",
        type=str,
        default="val-phase2",
        help="Prefix for validation data",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--test_prefix", type=str, default="test", help="Prefix for test data"
    )
    parser.add_argument(
        "--path_processed",
        type=str,
        default="../data/realdisp/",
        help="Path to processed data",
    )
    parser.add_argument("--lazy_load", action="store_false", help="Enable lazy loading")
    parser.add_argument("--use_weights", action="store_true", help="Use weights")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr_step", type=int, default=10, help="Step size for learning rate decay"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="Decay factor for learning rate"
    )
    parser.add_argument(
        "--weights_init",
        type=str,
        default="orthogonal",
        help="Weights initialization method",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3], help="Random seeds"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Augmentation configuration
    parser.add_argument(
        "--aug_first", action="store_false", help="Apply augmentation first"
    )
    parser.add_argument(
        "--magscale", action="store_true", help="Enable scaling augmentation"
    )
    parser.add_argument(
        "--magscale_mu", type=float, default=1.0, help="Scaling mean (mu)"
    )
    parser.add_argument(
        "--magscale_sigma",
        type=float,
        default=0.1,
        help="Scaling standard deviation (sigma)",
    )
    parser.add_argument(
        "--magscale_mean_center",
        action="store_true",
        help="Zero-center the scaling factor",
    )

    parser.add_argument(
        "--timescale", action="store_true", help="Enable time-scaling augmentation"
    )
    parser.add_argument(
        "--timescale_scale_min",
        type=float,
        default=0.8,
        help="TimeScaling scale_factor_min",
    )
    parser.add_argument(
        "--timescale_scale_max",
        type=float,
        default=1.2,
        help="TimeScaling scale_factor_max",
    )

    parser.add_argument(
        "--noisebias", action="store_true", help="Enable jittering augmentation"
    )
    parser.add_argument(
        "--noisebias_sigma",
        type=float,
        default=0.2,
        help="Noise standard deviation (sigma)",
    )
    parser.add_argument(
        "--noisebias_bias_min",
        type=float,
        default=-0.2,
        help="Bias min value",
    )
    parser.add_argument(
        "--noisebias_bias_max",
        type=float,
        default=0.2,
        help="Bias max value",
    )

    parser.add_argument(
        "--timewarp", action="store_true", help="Enable timewarp augmentation"
    )
    parser.add_argument(
        "--timewarp_sigma",
        type=float,
        default=0.1,
        help="TimeWarping standard deviation (sigma)",
    )
    parser.add_argument("--timewarp_knot", type=int, default=4, help="TimeWarping knot")
    parser.add_argument(
        "--timewarp_max_speed_ratio",
        type=float,
        default=1.5,
        help="TimeWarping max speed ratio",
    )

    parser.add_argument(
        "--magwarp", action="store_true", help="Enable magnitude warping augmentation"
    )
    parser.add_argument(
        "--magwarp_sigma",
        type=float,
        default=0.2,
        help="Magnitude warping standard deviation (sigma)",
    )
    parser.add_argument(
        "--magwarp_knot", type=int, default=4, help="Magnitude warping knot"
    )
    parser.add_argument(
        "--magwarp_mean_center",
        action="store_true",
        help="Zero-center the scaling factor",
    )

    # rot_range_x: torch.Tensor,  # torch.Tensor([min_angle, max_angle]),
    # rot_range_y: torch.Tensor,  # torch.Tensor([min_angle, max_angle]),
    # rot_range_z: torch.Tensor,  # torch.Tensor([min_angle, max_angle]),
    parser.add_argument(
        "--rotation", action="store_true", help="Enable rotation augmentation"
    )
    parser.add_argument(
        "--rotation_range_x",
        type=parse_range,
        default=(-180.0, 180.0),
        help="Rotation range for x-axis in the form 'min,max'",
    )

    parser.add_argument(
        "--rotation_range_y",
        type=parse_range,
        default=(-180.0, 180.0),
        help="Rotation range for y-axis in the form 'min,max'",
    )

    parser.add_argument(
        "--rotation_range_z",
        type=parse_range,
        default=(-180.0, 180.0),
        help="Rotation range for z-axis in the form 'min,max'",
    )

    parser.add_argument(
        "--wimusim_params_path",
        type=str,
        default=f"../data/realdisp/wimusim_params/",
        help="Path to WIMUSim parameters",
    )

    parser.add_argument(
        "--aug_prob",
        type=float,
        default=0.5,
        help="Probability of applying the augmentation",
    )
    parser.add_argument("--scenario", type=str, default="ideal", help="Scenario to use")

    args = parser.parse_args()

    # Convert args into a config dictionary similar to your current setup
    config = {
        "model": args.model,
        "window": args.window,
        "stride": args.stride,
        "dataset_name": args.dataset_name,
        "train-prefix": args.train_prefix,
        "val-prefix": args.val_prefix,
        "epochs": args.epochs,
        "test-prefix": args.test_prefix,
        "path_processed": args.path_processed,
        "lazy_load": args.lazy_load,
        "use_weights": args.use_weights,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_step": args.lr_step,
        "lr_decay": args.lr_decay,
        "weights_init": args.weights_init,
        "seeds": args.seeds,
        "verbose": args.verbose,
        "wimusim_params_path": args.wimusim_params_path,
        "scenario": args.scenario,
        "aug_config": {
            "aug_first": args.aug_first,
            "aug_prob": args.aug_prob,
            "magscale": None,  # Default to None if --magscale is not enabled
            "timescale": None,  # Default to None if --timescale is not enabled
            "noisebias": None,  # Default to None if --noisebias is not enabled
            "timewarp": None,  # Default to None if --timewarp is not enabled
            "magwarp": None,  # Default to None if --magwarp is not enabled
            "rotation": None,  # Default to None if --rotation is not enabled
        },
    }

    # Conditionally update scaling config if --scaling is enabled
    if args.magscale:
        config["aug_config"]["magscale"] = {
            "mu": args.magscale_mu,
            "sigma": args.magscale_sigma,
            "mean_center": args.magscale_mean_center,
        }

    if args.timescale:
        config["aug_config"]["timescale"] = {
            "scale_min": args.timescale_scale_min,
            "scale_max": args.timescale_scale_max,
        }

    if args.noisebias:
        config["aug_config"]["noisebias"] = {
            "noise_sigma": args.noisebias_sigma,
            "bias_min": args.noisebias_bias_min,
            "bias_max": args.noisebias_bias_max,
        }

    if args.timewarp:
        config["aug_config"]["timewarp"] = {
            "sigma": args.timewarp_sigma,
            "knot": args.timewarp_knot,
            "max_speed_ratio": args.timewarp_max_speed_ratio,
        }

    if args.magwarp:
        config["aug_config"]["magwarp"] = {
            "sigma": args.magwarp_sigma,
            "knot": args.magwarp_knot,
            "mean_center": args.magwarp_mean_center,
        }

    if args.rotation:
        config["aug_config"]["rotation"] = {
            "range_x": args.rotation_range_x,
            "range_y": args.rotation_range_y,
            "range_z": args.rotation_range_z,
        }

    return config


# main
if __name__ == "__main__":
    load_dotenv()
    # Use the fixed param for this experiment
    config = get_config()

    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "SPECIFY_YOUR_WANDB_ENTITY")

    if config["scenario"] == "ideal":
        WANDB_PROJECT = "simuaug_realdisp"
    else:
        WANDB_PROJECT = "simuaug_realdisp_self"
    randint = random.randint(0, 1000)
    aug_names = [
        key
        for key, value in config["aug_config"].items()
        if value is not None and key != "aug_first" and key != "aug_prob"
    ]
    if len(aug_names) > 1:
        warnings.warn(f"Multiple augmentation methods are enabled: {aug_names}. ")

    WANDB_RUN_NAME = f"realdisp_ppda_{aug_names[0]}_{randint}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={**config},
    )

    if config["scenario"] == "self":
        config["train-prefix"] = "train-self"
        config["val-prefix"] = "val-self"
        config["test-prefix"] = "test-self"

    print(paint(f"Applied Settings: "))
    print(json.dumps(config, indent=2, default=str))

    dataset_train = SensorDataset(
        config["dataset_name"],
        window=config["window"],
        stride=config["stride"],
        prefix=config["train-prefix"],
        path_processed=config["path_processed"],
        lazy_load=config["lazy_load"],
    )

    # Use the mean and std from the training dataset
    scaling_config = {"mean": dataset_train.mean, "std": dataset_train.std}

    dataset_val = SensorDataset(
        config["dataset_name"],
        window=config["window"],
        stride=config["stride"],
        prefix=config["val-prefix"],
        path_processed=config["path_processed"],
        lazy_load=config["lazy_load"],
        **scaling_config,
    )

    dataset_test = SensorDataset(
        config["dataset_name"],
        window=config["window"],
        stride=config["stride"],
        prefix=config["test-prefix"],
        path_processed=config["path_processed"],
        lazy_load=config["lazy_load"],
        **scaling_config,
    )

    B_list, P_list, D_list, H_list = [], [], [], []
    groups = []  # Used to control the sampling process
    activity_name = []
    target_list = []

    # Only load the first 10 subjects (will be modified when testing different splits)
    if config["scenario"] == "ideal":
        subject_ids = range(1, 11)
    else:
        # For self scenario, use all the subjects in ideal scenario for training.
        subject_ids = range(1, 18)

    for subject_id in subject_ids:
        pkl_file_path = f"{config['wimusim_params_path']}/realdisp_ideal_p{subject_id:03d}_wimusim_params_25Hz.pkl"
        with open(pkl_file_path, "rb") as f:
            wimusim_params_dict = pickle.load(f)
            B_list.append(wimusim_params_dict["B"])
            D_list.append(wimusim_params_dict["D"])
            P_list.append(wimusim_params_dict["P"])
            H_list.append(wimusim_params_dict["H"])
            target_list.append(wimusim_params_dict["target"])

    wimusim_dataset = WIMUSimDataset(
        B_list=B_list,
        D_list=D_list,
        P_list=P_list,
        H_list=H_list,
        target_list=target_list,
        window=100,  # 1 seconds
        stride=25,  # 0.25 seconds
        acc_only=False,
        scale_config=scaling_config,
    )

    device = get_default_device()
    identity = Identity(device=device)
    aug_prob = config["aug_config"]["aug_prob"]
    if config["aug_config"].get("magscale", None) is not None:
        mu = torch.tensor(
            config["aug_config"]["magscale"]["mu"], requires_grad=False, device=device
        )
        log_sigma = torch.tensor(
            np.log(config["aug_config"]["magscale"]["sigma"]),
            requires_grad=False,
            device=device,
        )
        magscale = MagnitudeScaling(
            mu,
            log_sigma,
            joint_names=["PELVIS", "R_SHOULDER", "R_ELBOW", "L_SHOULDER", "L_ELBOW"],
            mean_center=config["aug_config"]["magscale"]["mean_center"],
            # Use the following for the REALWORLD dataset
            # joint_names=["PELVIS", "R_SHOULDER", "R_ELBOW", "L_SHOULDER", "L_ELBOW", "R_HIP", "L_HIP",
            # "R_KNEE", "L_KNEE"],
        )
        sub_policies = [[identity], [magscale]]
        a_subpolicies = torch.tensor(
            [1 - aug_prob, aug_prob], device=device, requires_grad=True
        )
        aug_params = [a_subpolicies, log_sigma]
    elif config["aug_config"].get("timescale", None) is not None:
        scale_min = float(config["aug_config"]["timescale"]["scale_min"])
        scale_max = float(config["aug_config"]["timescale"]["scale_max"])
        timescale = TimeScaling(
            scale_factor_min=scale_min, scale_factor_max=scale_max, device=device
        )
        sub_policies = [[identity], [timescale]]
        a_subpolicies = torch.tensor(
            [1 - aug_prob, aug_prob], device=device, requires_grad=True
        )
        aug_params = [a_subpolicies]
    elif config["aug_config"].get("noisebias", None) is not None:
        noise_bias = NoiseBias(
            noise_sigma=config["aug_config"]["noisebias"]["noise_sigma"],
            bias_min=config["aug_config"]["noisebias"]["bias_min"],
            bias_max=config["aug_config"]["noisebias"]["bias_max"],
            device=device,
        )
        sub_policies = [[identity], [noise_bias]]
        a_subpolicies = torch.tensor(
            [1 - aug_prob, aug_prob], device=device, requires_grad=True
        )
        aug_params = [a_subpolicies]
    elif config["aug_config"].get("timewarp", None) is not None:
        sigma = config["aug_config"]["timewarp"]["sigma"]
        knot = config["aug_config"]["timewarp"]["knot"]
        max_speed_ratio = config["aug_config"]["timewarp"]["max_speed_ratio"]
        timewarp = TimeWarping(
            sigma=sigma, knot=knot, max_speed_ratio=max_speed_ratio, device=device
        )
        sub_policies = [[identity], [timewarp]]
        a_subpolicies = torch.tensor(
            [1 - aug_prob, aug_prob], device=device, requires_grad=True
        )
        aug_params = [a_subpolicies]
    elif config["aug_config"].get("magwarp", None) is not None:
        sigma = config["aug_config"]["magwarp"]["sigma"]
        knot = config["aug_config"]["magwarp"]["knot"]
        mean_center = config["aug_config"]["magwarp"]["mean_center"]
        magwarp = MagnitudeWarping(
            sigma=sigma, knot=knot, mean_center=mean_center, device=device
        )
        sub_policies = [[identity], [magwarp]]
        a_subpolicies = torch.tensor(
            [1 - aug_prob, aug_prob], device=device, requires_grad=True
        )
        aug_params = [a_subpolicies]

    elif config["aug_config"].get("rotation", None) is not None:
        rot_range_x = torch.tensor(
            config["aug_config"]["rotation"]["range_x"], device=device
        )
        rot_range_y = torch.tensor(
            config["aug_config"]["rotation"]["range_y"], device=device
        )
        rot_range_z = torch.tensor(
            config["aug_config"]["rotation"]["range_z"], device=device
        )
        rotation = Rotation(rot_range_x, rot_range_y, rot_range_z, device=device)
        sub_policies = [[identity], [rotation]]
        a_subpolicies = torch.tensor(
            [1 - aug_prob, aug_prob], device=device, requires_grad=True
        )
        if config["scenario"] == "self":
            rot_range_x = torch.tensor((-90, 90), device=device)
            rot_range_y = torch.tensor((-0, 0), device=device)
            rot_range_z = torch.tensor((-0, 0), device=device)
            rotation_wrist = Rotation(
                rot_range_x, rot_range_y, rot_range_z, device=device
            )
            sub_policies.append([rotation_wrist])

            rot_range_x = torch.tensor((179, 180), device=device)
            rot_range_y = torch.tensor((-0, 0), device=device)
            rot_range_z = torch.tensor((-0, 0), device=device)
            rotation_flip_x = Rotation(
                rot_range_x, rot_range_y, rot_range_z, device=device
            )
            sub_policies.append([rotation_flip_x])

            rot_range_x = torch.tensor((-0, 0), device=device)
            rot_range_y = torch.tensor((179, 180), device=device)
            rot_range_z = torch.tensor((-0, 0), device=device)
            rotation_flip_y = Rotation(
                rot_range_x, rot_range_y, rot_range_z, device=device
            )
            sub_policies.append([rotation_flip_y])

            rot_range_x = torch.tensor((-0, 0), device=device)
            rot_range_y = torch.tensor((-0, 0), device=device)
            rot_range_z = torch.tensor((179, 180), device=device)
            rotation_flip_z = Rotation(
                rot_range_x, rot_range_y, rot_range_z, device=device
            )
            sub_policies.append([rotation_flip_z])
            # Use different a_subpolicies for this scenario.
            a_subpolicies = torch.tensor(
                [2.0, 2.0, 0.001, 0.001, 0.001, 0.001],
                # sigmoid() => [0.3934, 0.3934, 0.0533, 0.0533, 0.0533, 0.0533]
                device=device,
                requires_grad=True,
            )

        aug_params = [a_subpolicies]

    else:
        # TODO: define other augmentation methods
        sub_policies = [[Identity(device=device)]]  # No augmentation
        a_subpolicies = torch.tensor([1.0], device=device, requires_grad=True)
        aug_params = [a_subpolicies]

    aug_planner = AugPolicyPlanner(
        sub_policies=sub_policies,
        aug_params=aug_params,
        a_subpolicies=a_subpolicies,
        tau=1.0,
        device=device,
    )

    if config["use_weights"]:
        class_weights = (
            torch.from_numpy(
                class_weight.compute_class_weight(
                    "balanced",
                    classes=np.unique(dataset_train.target),
                    y=dataset_train.target,
                )
            )
            .float()
            .cuda()
        )
    else:
        # Use this for this experiment to align with the previous experiments
        class_weights = None

    train_results_list = []
    test_results_list = []
    start_time = time.time()

    log_date = time.strftime("%Y%m%d")
    log_timestamp = time.strftime("%H%M%S")

    if config["model"] == "DeepConvLSTM":
        model = DeepConvLSTM(
            n_channels=dataset_train.n_channels,
            n_classes=dataset_train.n_classes,
            dataset="realdisp",
            experiment=f"/{log_date}/{log_timestamp}",
        ).cuda()
    else:
        model = AttendAndDiscriminate(
            input_dim=dataset_train.n_channels,
            num_class=dataset_train.n_classes,
            dataset="realdisp",
            experiment=f"/{log_date}/{log_timestamp}",
        ).cuda()

    print(
        paint(f"Running HAR training & evaluation loop with seeds {config['seeds']}...")
    )
    base_path_checkpoints = model.path_checkpoints
    for seed in config["seeds"]:
        print(paint("Running with random seed set to {0}...".format(str(seed))))
        seed_torch(seed)  # Set the seed
        model.path_checkpoints = (
            base_path_checkpoints + f"/seed_{seed}"
        )  # Update the path to save the checkpoints

        # Phase 1: Train the model while tuning the augmentation parameters
        (
            t_loss,
            t_acc,
            t_fm,
            t_fw,
            v_loss,
            v_acc,
            v_fm,
            v_fw,
            criterion,
        ) = train_model_with_ppda(
            model=model,
            train_data=dataset_train,
            train_data_sim=wimusim_dataset,
            val_data=dataset_val,
            test_data=dataset_test,
            lr=config["lr"],
            class_weights=class_weights,
            seed=seed,
            n_epochs=config["epochs"],
            batch_size_train=config["batch_size"],
            aug_planner=aug_planner,
            aug_first=config["aug_config"]["aug_first"],
        )
        loss_test, acc_test, fm_test, fw_test, elapsed = eval_model(
            model, dataset_test, batch_size=256, seed=seed
        )

        # Save the results
        results_row = {
            "v_type": "split",
            "seed": seed,
            "sbj": -1,
            "t_loss": t_loss,
            "t_acc": t_acc,
            "t_fm": t_fm,
            "t_fw": t_fw,
            "v_loss": v_loss,
            "v_acc": v_acc,
            "v_fm": v_fm,
            "v_fw": v_fw,
        }

        tests_results_row = {
            "v_type": "split",
            "seed": seed,
            "test_loss": loss_test,
            "test_acc": acc_test,
            "test_fm": fm_test,
            "test_fw": fw_test,
        }

        train_results_list.append(results_row)
        test_results_list.append(tests_results_row)

    # After the loop, convert lists of dictionaries to DataFrames
    train_results = pd.DataFrame(train_results_list)
    test_results = pd.DataFrame(test_results_list)

    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))

    print(paint(f"Finished HAR training loop (h:m:s): {elapsed}"))
    print(paint("--" * 75, "blue"))

    run_train_analysis(train_results)
    run_test_analysis(test_results)

    wandb_logging(train_results, test_results, {**config})
    wandb.finish()
