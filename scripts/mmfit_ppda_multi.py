import os
import sys
import json
import argparse

sys.path.append("..")

import time
from datetime import timedelta
import random
import pandas as pd
import wandb
import glob
import pickle
from itertools import product
from dotenv import load_dotenv

import numpy as np
from sklearn.utils import class_weight

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
from models.utils import init_weights
from scripts.utils import eval_model, eval_one_epoch, run_test_analysis, wandb_logging, train_one_epoch_phase1
from simuaug.dataloaders import PPDADataLoader, STDADataLoader
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
    tau=5.0,
    n_batches_per_epoch=None,
    early_stopping_patience=20,
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
        worker_init_fn=np.random.seed(int(seed)),
        aug_planner=aug_planner,
        sample=True,
        n_batches_per_epoch=n_batches_per_epoch,
    )

    loader_val = DataLoader(
        val_data,
        batch_size=batch_size_train,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
    )

    loader_val_aug = STDADataLoader(
        val_data,
        batch_size=batch_size_train,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
        aug_planner=aug_planner,
        sample=False,
        eval=True,
    )

    loader_test = DataLoader(
        test_data,
        batch_size=batch_size_test,
        shuffle=False,
        worker_init_fn=np.random.seed(int(seed)),
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer_m = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_a = torch.optim.Adam(
        [a_subpolicies, log_sigma], lr=lr_a, betas=(0.5, 0.999), weight_decay=1e-3
    )  #
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_m, step_size=lr_step, gamma=lr_decay
    )

    init_weights(model, weights_init)
    aug_planner.reinit_a_subpolicies(1e-3)
    aug_planner.init_tau(tau)

    metric_best = 0.0
    early_stopping_counter = 0

    # Store the training and validation metrics
    t_loss, t_acc, t_fm, t_fw = [], [], [], []
    v_loss, v_acc, v_fm, v_fw = [], [], [], []

    for epoch in range(n_epochs):
        if verbose:
            print(
                f"------------------------------ Epoch {epoch + 1}/{n_epochs} ------------------------------\n"
                f"Learning rate: {optimizer_m.param_groups[0]['lr']}, tau: {aug_planner.tau}"
            )
        aug_planner.update_tau(epoch)

        train_one_epoch_phase1(
            model=model,
            loader_train=loader_train_aug,
            loader_val=loader_val_aug,
            criterion=criterion,
            optimizer_m=optimizer_m,
            optimizer_a=optimizer_a,
            aug_planner=aug_planner,
        )

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
            "optim_state_dict": optimizer_m.state_dict(),
            "optim_a_state_dict": optimizer_a.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:  # ignore first 5 epochs
            # Don't update the best metric for the first 5 epochs as it's unstable
            metric_best = metric
            early_stopping_counter = 0  # Reset early stopping counter
            if verbose:
                print(
                    paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue")
                )
            torch.save(checkpoint, os.path.join(model.path_checkpoints, "best.pth"))

        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            if verbose:
                print(
                    paint(
                        f"Early stopping triggered after {early_stopping_patience} epochs without improvement.",
                        "red",
                    )
                )
            break  # Exit the training loop
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
    parser.add_argument("--window", type=int, default=60, help="Window size")
    parser.add_argument("--stride", type=int, default=30, help="Stride size")
    parser.add_argument(
        "--dataset_name", type=str, default="mmfit", help="Name of the dataset"
    )
    parser.add_argument(
        "--val_prefix",
        type=str,
        default="val",
        help="Prefix for validation data",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--test_prefix", type=str, default="test", help="Prefix for test data"
    )
    parser.add_argument(
        "--path_processed",
        type=str,
        default="../data/mmfit/",
        help="Path to processed data",
    )
    parser.add_argument("--lazy_load", action="store_false", help="Enable lazy loading")
    parser.add_argument("--use_weights", action="store_true", help="Use weights")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr_step", type=int, default=10, help="Step size for learning rate decay"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.9, help="Decay factor for learning rate"
    )
    parser.add_argument("--tau", type=float, default=5.0, help="Learning rate")

    parser.add_argument(
        "--weights_init",
        type=str,
        default="orthogonal",
        help="Weights initialization method",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1], help="Random seeds"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    parser.add_argument(
        "--wimusim_params_path",
        type=str,
        default=f"../data/mmfit/wimusim_params/",
        help="Path to WIMUSim parameters",
    )

    parser.add_argument(
        "--n_sub", type=int, default=1, help="Number of subjects to use"
    )
    parser.add_argument(
        "--comb_idx", type=int, default=0, help="Combination index (0 to 9)"
    )
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument(
        "--paramix", action="store_true", help="Enable paramix for P params"
    )
    parser.add_argument(
        "--scale_translation", action="store_true", help="Scale translation"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=100,
        help="Early stopping patience.",
    )
    args = parser.parse_args()

    # Convert args into a config dictionary similar to your current setup
    config = {
        "model": args.model,
        "window": args.window,
        "stride": args.stride,
        "dataset_name": args.dataset_name,
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
        "tau": args.tau,
        "weights_init": args.weights_init,
        "seeds": args.seeds,
        "verbose": args.verbose,
        "wimusim_params_path": args.wimusim_params_path,
        "n_sub": args.n_sub,
        "comb_idx": args.comb_idx,
        "offline": args.offline,
        "paramix": args.paramix,
        "scale_translation": args.scale_translation,
        "early_stopping_patience": args.early_stopping_patience,
    }

    return config


# main
if __name__ == "__main__":
    # Use the fixed param for this experiment
    config = get_config()
    load_dotenv()

    # set the train_prefix based on the given n_sub and comb_idx
    combination_dict_path = "../data/mmfit/n_sub_combinations_dict.pkl"
    with open(combination_dict_path, "rb") as f:
        combination_dict = pickle.load(f)

    recording_ids = combination_dict[config["n_sub"]][config["comb_idx"]]
    config["train-prefix"] = [f"train-{sub_id}" for sub_id in recording_ids]
    print(config["train-prefix"])

    WANDB_PROJECT = "simuaug_mmfit_n_sub"
    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "SPECIFY_YOUR_WANDB_ENTITY")
    randint = random.randint(0, 1000)
    WANDB_RUN_NAME = f"mmfit_ppda_{config['n_sub']}_{config['comb_idx']}_{randint}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        mode="offline" if config["offline"] else "online",
        config={**config},
    )

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

    # Only load the first 13 subjects
    for recording_id in recording_ids:
        sbj_pkl_files = glob.glob(
            f"{config['wimusim_params_path']}/w{recording_id:02d}_wimusim_params_dict.pkl"
        )
        if len(sbj_pkl_files) != 1:
            raise ValueError(
                f"Invalid number of WIMUSim parameter files for subject {recording_id}."
            )
        for pkl_file_path in sbj_pkl_files:
            print("Loading WIMUSim parameters from: ", pkl_file_path)
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
        window=60,  # 2 seconds
        stride=30,  # 1.0 seconds
        acc_only=False,
        scale_config=scaling_config,
    )

    device = get_default_device()
    identity = Identity(device=device)

    group1 = [identity]
    group2 = [identity]
    group3 = [identity]
    group4 = [identity]

    # Add MagScales
    for i, sigma in enumerate([0.1, 0.2, 0.4, 0.6]):
        mu = torch.tensor(1.0, requires_grad=False, device=device)
        log_sigma = torch.tensor(
            np.log(sigma),
            requires_grad=False,
            device=device,
        )
        magscale = MagnitudeScaling(
            mu,
            log_sigma,
            joint_names=[
                "NECK",
                "R_SHOULDER",
                "L_SHOULDER",
            ],
            mean_center=False,
            scale_translation=config["scale_translation"],
            device=device,
        )
        group1.append(magscale)
    for i, (knot, sigma) in enumerate([(2, 0.2), (2, 0.4), (4, 0.2), (4, 0.4)]):
        magwarp = MagnitudeWarping(
            sigma=sigma,
            knot=knot,
            mean_center=True,
            joint_idx=[10, 15, 18],
            scale_translation=config["scale_translation"],
            device=device,
        )
        group1.append(magwarp)
    for i, (scale_min, scale_max) in enumerate(
        [(0.7, 0.9), (1.1, 1.3), (0.75, 1.5), (0.5, 2.0)]
    ):
        timescale = TimeScaling(
            scale_factor_min=scale_min, scale_factor_max=scale_max, device=device
        )
        group2.append(timescale)
    for i, (knot, max_speed_ratio) in enumerate(
        [(2, 1.5), (2, 2.0), (4, 1.5), (4, 2.0)]
    ):
        sigma = 0.1
        timewarp = TimeWarping(
            sigma=sigma, knot=knot, max_speed_ratio=max_speed_ratio, device=device
        )
        group2.append(timewarp)
    for i, (range_x, range_y, range_z) in enumerate(
        [
            [(-25, 25), (-25, 25), (-25, 25)],
            # [(-180, 180), (-180, 180), (-180, 180)],
        ]
    ):
        rot_range_x = torch.tensor(range_x, device=device)
        rot_range_y = torch.tensor(range_y, device=device)
        rot_range_z = torch.tensor(range_z, device=device)
        # Enable flipping for realdisp with 25% probability
        rotation = Rotation(
            rot_range_x,
            rot_range_y,
            rot_range_z,
            flip_prob=0.0,
            paramix=config["paramix"],
            device=device,
        )
        group3.append(rotation)
    for i, sigma in enumerate([0.05, 0.10, 0.15, 0.20]):
        noise_bias = NoiseBias(
            noise_sigma=sigma, bias_min=-1.0, bias_max=1.0, device=device
        )
        group4.append(noise_bias)

    sub_policies = list(product(group1, group2, group3, group4))
    a_subpolicies = torch.tensor(
        [1e-3] * len(sub_policies), device=device, requires_grad=True
    )

    aug_planner = AugPolicyPlanner(
        sub_policies=sub_policies,
        aug_params=[a_subpolicies],
        a_subpolicies=a_subpolicies,
        tau=config["tau"],
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
            dataset="mmfit",
            experiment=f"/{log_date}/{log_timestamp}",
        ).cuda()
    else:
        model = AttendAndDiscriminate(
            input_dim=dataset_train.n_channels,
            num_class=dataset_train.n_classes,
            dataset="mmfit",
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

        if config["dataset_name"] == "mmfit":
            n_batches_per_epoch = 108
        else:
            raise NotImplementedError

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
            n_batches_per_epoch=n_batches_per_epoch,
            tau=config["tau"],
            early_stopping_patience=config["early_stopping_patience"],
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

    run_test_analysis(test_results)

    wandb_logging(train_results, test_results, {**config})
    wandb.finish()
