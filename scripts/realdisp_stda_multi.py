import json
import os
import pickle
import sys
import time
from datetime import timedelta
import random
import argparse
import pandas as pd
import wandb

sys.path.append("..")
from itertools import product

import numpy as np
from sklearn.utils import class_weight

import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from simuaug.utils import get_default_device, paint, seed_torch
from simuaug.datasets import SensorDataset
from simuaug.augmentations import AugPolicyPlanner
from simuaug.augmentations.stda import (
    Identity,
    MagnitudeScaling,
    MagnitudeWarping,
    TimeScaling,
    TimeWarping,
    Rotation,
    Jittering,
)

from models import DeepConvLSTM, AttendAndDiscriminate
from models.utils import (
    init_weights,
    wandb_logging,
    AverageMeter,
    cycle_loader,
    calc_epsilon,
    set_model_params,
)
from models.utils import run_train_analysis, run_test_analysis

from scripts.realdisp_baseline import eval_model, eval_one_epoch
from scripts.realdisp_stda_scaling_opt import train_model
from simuaug.dataloaders import STDADataLoader


def train_one_epoch_phase1(
    model,
    loader_train,
    loader_val,
    criterion,
    optimizer_m,
    optimizer_a,
    verbose=True,
    print_freq=100,
    aug_planner=None,
):
    """
    Train the model for one epoch. Updated to include logging of augmentation statistics.

    :param model:
    :param loader_train:
    :param loader_val:
    :param criterion:
    :param optimizer_m: Optimizer for the model parameters
    :param optimizer_a: Optimizer for the augmentation parameters
    :param verbose:
    :param print_freq:
    :return:
    """
    losses = AverageMeter("Loss")
    model.train()

    log_aug_list = []

    if len(loader_train) > len(loader_val):
        loader_train_iter = loader_train  # Repeat validation loader
        loader_val_iter = cycle_loader(loader_val)
    else:
        loader_train_iter = cycle_loader(loader_train)  # Repeat train loader
        loader_val_iter = loader_val

    # for batch_idx, (data, target, idx) in enumerate(loader):
    for batch_idx, (
        (data_train, target_train, idx_train),
        (data_val, target_val, idx_val),
    ) in enumerate(zip(loader_train_iter, loader_val_iter)):
        data_train = data_train.cuda()
        data_val = data_val.cuda()
        target_train = target_train.view(-1).cuda()
        target_val = target_val.view(-1).cuda()

        ## Step 1. Update model weights
        optimizer_m.zero_grad()
        optimizer_a.zero_grad()
        z, logits = model(data_train)
        loss_train = criterion(logits, target_train)
        losses.update(loss_train.item(), data_train.shape[0])

        loss_train.backward(retain_graph=True)
        optimizer_m.step()  # Only update the model weights

        # Log the current subpolicy
        log_aug_list.append(
            loader_train.aug_planner.current_subpolicy_one_hot.detach().cpu().numpy()
        )

        ## Step 2. Update aug params
        optimizer_m.zero_grad()
        optimizer_a.zero_grad()

        z, logits = model(data_val)
        loss_val = criterion(
            logits, target_val
        )  # \nabla_{\alpha, \hp} the left term in (11)

        grad_w_val = torch.autograd.grad(
            loss_val, model.parameters(), create_graph=True
        )
        eps = calc_epsilon(grad_w_val)

        grad_aug_val = torch.autograd.grad(
            loss_val, aug_planner.aug_params, retain_graph=True, allow_unused=True
        )

        # Keep the original parameters
        w_orig = [p.clone() for p in model.parameters()]

        # Step 2.1: Compute w_plus
        optimizer_m.zero_grad()
        optimizer_a.zero_grad()
        w_plus = [p + eps * g for p, g in zip(model.parameters(), grad_w_val)]
        set_model_params(model, w_plus)
        z, logits = model(data_train)
        loss_plus = criterion(logits, target_train)
        grad_aug_plus = torch.autograd.grad(
            loss_plus, aug_planner.aug_params, retain_graph=True, allow_unused=True
        )

        # Step 2.2: Compute w_minus
        optimizer_m.zero_grad()
        optimizer_a.zero_grad()
        set_model_params(model, w_orig)
        w_minus = [p - eps * g for p, g in zip(model.parameters(), grad_w_val)]
        set_model_params(model, w_minus)

        z, logits = model(data_train)
        loss_minus = criterion(logits, target_train)
        grad_aug_minus = torch.autograd.grad(
            loss_minus, aug_planner.aug_params, allow_unused=True
        )

        if verbose:
            if batch_idx % print_freq == 0:
                print(
                    f"[-] Batch {batch_idx + 1}/{len(loader_train)}\t Loss: {str(losses)}"
                )
                print(
                    f"[-] eps: {eps.item():.6f}, loss_plus: {loss_plus.item():.3f}, loss_minus: {loss_minus.item():3f}"
                )
        # Update the augmentation parameters
        # Used subpolicies are not the same for train and val
        for aug_param, grad_val, grad_plus, grad_minus in zip(
            aug_planner.aug_params, grad_aug_val, grad_aug_plus, grad_aug_minus
        ):
            if grad_val is None:
                aug_param.grad = None
            else:
                # Corresponds to the equation (11) in AutoAugHAR
                aug_param.grad = grad_val - (grad_plus - grad_minus) / (2 * eps)

        # print(a_subpolicies.grad, log_sigma.grad)
        optimizer_a.step()

        # Set the model parameters back to the original values
        set_model_params(model, w_orig)
        # optimizer_m.zero_grad()
        # optimizer_a.zero_grad()

    # Print the augmentation statistics
    # Convert log_aug_list to a numpy array for easier summation
    log_aug_array = np.array(log_aug_list)
    # Sum across all batches to get absolute numbers for each augmentation
    total_aug_counts = np.sum(log_aug_array, axis=0)
    # Calculate the fractions (percentages) of each augmentation applied
    total_lengths = len(log_aug_array)
    aug_fractions = total_aug_counts / total_lengths

    # Apply sigmoid to a_subpolicies
    p_subpolicies = torch.nn.functional.softmax(aug_planner.a_subpolicies, dim=0)
    # Normalize to ensure the probabilities sum to 1
    p_subpolicies_formatted = ", ".join(
        [f"{prob:.4f}" for prob in p_subpolicies.detach().cpu().numpy()]
    )
    aug_fractions_formatted = ", ".join([f"{frac:.4f}" for frac in aug_fractions])
    # Print the absolute counts and fractions
    print(f"[-] Aug stats: p_subpolicies: {p_subpolicies_formatted}")
    print(
        f"[-] Aug stats: counts: {total_aug_counts}, ratios: {aug_fractions_formatted}"
    )


def train_model_phase_1(
    model,
    train_data,
    val_data,
    test_data,
    batch_size_train,
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
    aug_planner=None,
    tau=5,
    n_batches_per_epoch=None,
):
    if verbose:
        print(
            paint(
                f"================= Running HAR training loop with seed {seed} ================="
            )
        )

    loader_train = STDADataLoader(
        train_data,
        batch_size=batch_size_train,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
        aug_planner=aug_planner,
        sample=True,
        n_batches_per_epoch=n_batches_per_epoch,
    )

    # Use the same augmentation planner for the validation set (training phase)
    loader_val_aug = STDADataLoader(
        val_data,
        batch_size=batch_size_train,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
        aug_planner=aug_planner,
        sample=False,
        eval=True,
    )

    # Use this loader for the validation set (validation phase)
    loader_val = DataLoader(
        val_data,
        batch_size=batch_size_train,
        shuffle=False,
        worker_init_fn=np.random.seed(int(seed)),
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
            loader_train=loader_train,
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
        if metric >= metric_best:  # Ignore first 5 epochs
            if verbose:
                print(
                    paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue")
                )
            # Don't update the best metric for the first 5 epochs as it's unstable
            metric_best = metric
            torch.save(checkpoint, os.path.join(model.path_checkpoints, "best.pth"))

            # Save the a_subpolicies
            torch.save(
                a_subpolicies, os.path.join(model.path_checkpoints, "a_subpolicies.pth")
            )

        if lr_step > 0:
            scheduler.step()

    return t_loss, t_acc, t_fm, t_fw, v_loss, v_acc, v_fm, v_fw, criterion


def print_augmentation_statistics(aug_planner):
    a_subpolicies_sigmoid = torch.sigmoid(aug_planner.a_subpolicies)

    # Normalize to ensure the probabilities sum to 1
    p_subpolicies = a_subpolicies_sigmoid / a_subpolicies_sigmoid.sum()
    p_subpolicies_formatted = ", ".join(
        [f"{prob:.4f}" for prob in p_subpolicies.detach().cpu().numpy()]
    )
    # Print the absolute counts and fractions
    print(paint(f"[-] Aug stats: p_subpolicies: {p_subpolicies_formatted}"))


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
        default="train-ideal",
        help="Prefix for training data",
    )
    parser.add_argument(
        "--val_prefix",
        type=str,
        default="val-ideal",
        help="Prefix for training data",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--test_prefix", type=str, default="test-ideal", help="Prefix for test data"
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
        "--lr_a", type=float, default=5 * 1e-3, help="Learning rate for a_subpolicies"
    )
    parser.add_argument("--tau", type=float, default=5, help="tau for a_subpolicies")
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

    parser.add_argument(
        "--n_sub", type=int, default=1, help="Number of subjects to use"
    )
    parser.add_argument(
        "--comb_idx", type=int, default=0, help="Combination index (0 to 9)"
    )
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")

    args = parser.parse_args()

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
        "lr_a": args.lr_a,
        "tau": args.tau,
        "lr_step": args.lr_step,
        "lr_decay": args.lr_decay,
        "weights_init": args.weights_init,
        "seeds": args.seeds,
        "verbose": args.verbose,
        "n_sub": args.n_sub,
        "comb_idx": args.comb_idx,
        "offline": args.offline,
    }

    return config


# main
if __name__ == "__main__":
    load_dotenv()
    # Use the fixed param for this experiment
    config = get_config()
    # set the train_prefix based on the given n_sub and comb_idx
    combination_dict_path = "../data/realdisp/n_sub_combinations_dict.pkl"
    with open(combination_dict_path, "rb") as f:
        combination_dict = pickle.load(f)

    subject_ids = combination_dict[config["n_sub"]][config["comb_idx"]]
    config["train-prefix"] = [f"train-{sub_id}" for sub_id in subject_ids]
    print(config["train-prefix"])
    WANDB_PROJECT = "simuaug_realdisp_n_sub"

    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "SPECIFY_YOUR_WANDB_ENTITY")

    randint = random.randint(0, 1000)

    WANDB_RUN_NAME = f"realdisp_stda_{config['n_sub']}_{config['comb_idx']}_{randint}"

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

    device = get_default_device()

    # Define data augmentation policies
    # For now, let's start with only one data augmentation policy
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
        magscale = MagnitudeScaling(mu, log_sigma)
        group1.append(magscale)

    for i, (knot, sigma) in enumerate([(2, 0.2), (2, 0.4), (4, 0.2), (4, 0.4)]):
        magwarp = MagnitudeWarping(sigma=sigma, knot=knot, device=device)
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
            [(-180, 180), (-180, 180), (-180, 180)],
        ]
    ):
        rot_range_x = torch.tensor(range_x, device=device)
        rot_range_y = torch.tensor(range_y, device=device)
        rot_range_z = torch.tensor(range_z, device=device)
        rotation = Rotation(rot_range_x, rot_range_y, rot_range_z, device=device)
        group3.append(rotation)

    for i, sigma in enumerate([0.05, 0.10, 0.15, 0.20]):
        log_sigma = torch.tensor(
            np.log(sigma),
            requires_grad=False,
            device=device,
        )
        jitter = Jittering(log_sigma, device=device)
        group4.append(jitter)

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

    # sys.stdout = Logger(os.path.join(model.path_logs, "log"))

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
        ) = train_model_phase_1(
            model,
            dataset_train,
            dataset_val,
            dataset_test,
            lr=config["lr"],
            lr_a=config["lr_a"],
            class_weights=class_weights,
            seed=seed,
            n_epochs=config["epochs"],
            batch_size_train=config["batch_size"],
            aug_planner=aug_planner,
            tau=config["tau"],
            n_batches_per_epoch=118,
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
