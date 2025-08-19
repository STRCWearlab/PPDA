import os
import sys
import time
from datetime import timedelta
import random

sys.path.append("..")

import pandas as pd
import wandb

import numpy as np
from sklearn.utils import class_weight
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

from simuaug.utils import get_default_device, paint, seed_torch
from simuaug.datasets import SensorDataset

from models import DeepConvLSTM, AttendAndDiscriminate
from models.utils import init_weights, Logger

from scripts.utils import eval_model, eval_one_epoch, train_one_epoch_baseline

import argparse
from scripts.utils import wandb_logging, run_test_analysis

def train_model(
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
    lr_step=10,
    lr_decay=0.9,
    weights_init="orthogonal",
    batch_size_test=256,
    early_stopping_patience=100,
):
    if verbose:
        print(
            paint(
                f"================= Running HAR training loop with seed {seed} ================="
            )
        )

    loader_train = DataLoader(
        train_data,
        batch_size=batch_size_train,
        shuffle=True,
        worker_init_fn=np.random.seed(int(seed)),
    )
    loader_val = DataLoader(
        val_data,
        batch_size=batch_size_test,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_decay
    )

    init_weights(model, weights_init)

    metric_best = 0.0
    early_stopping_counter = 0  # Counter for early stopping

    # Store the training and validation metrics
    t_loss, t_acc, t_fm, t_fw = [], [], [], []
    v_loss, v_acc, v_fm, v_fw = [], [], [], []

    for epoch in range(n_epochs):
        if verbose:
            print(
                f"------------------------------ Epoch {epoch + 1}/{n_epochs} ------------------------------\n"
                f"Learning rate: {optimizer.param_groups[0]['lr']}"
            )
        train_one_epoch_baseline(model, loader_train, criterion, optimizer)

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
            # Don't update the best metric for the first 5 epochs as it's unstable
            if verbose:
                print(
                    paint(
                        f"[*] Saving checkpoint... ({metric_best}->{metric})",
                        "blue",
                    )
                )
            # Don't update the best metric for the first 5 epochs as it's unstable
            metric_best = metric
            early_stopping_counter = 0  # Reset early stopping counter
            # Don't update the best metric for the first 5 epochs as it's unstable
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


def get_config():
    parser = argparse.ArgumentParser(description="Configuration for experiment")

    # Adding all the parameters from your config
    parser.add_argument(
        "--model", type=str, default="DeepConvLSTM", help="DeepLearning model"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--use_weights", action="store_true", help="Use weighted loss for training"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=100,
        help="Early stopping patience.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--dataset_name", type=str, default="realworld", help="Dataset name"
    )
    args = parser.parse_args()
    config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "use_weights": args.use_weights,
        "early_stopping_patience": args.early_stopping_patience,
        "lr": args.lr,
        "dataset_name": args.dataset_name,
    }

    return config


if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    sys.path.append("..")
    config = get_config()

    config = {
        "model": config["model"],
        "window": 30,
        "stride": 15,
        "dataset_name": config["dataset_name"],
        "train-prefix": "train",
        "val-prefix": "val",
        "test-prefix": "test",
        "path_processed": f"../data/{config['dataset_name']}/",
        "lazy_load": True,
        "use_weights": config["use_weights"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "lr_step": 10,
        "lr_decay": 0.9,
        "weights_init": "orthogonal",
        "seeds": [1, 2, 3],
        "epochs": config["epochs"],
        "verbose": True,
        "early_stopping_patience": config["early_stopping_patience"],
    }

    device = get_default_device()

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

    WANDB_PROJECT = f"simuaug_{config['dataset_name']}"
    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "SPECIFY_YOUR_WANDB_ENTITY")

    WANDB_RUN_NAME = f"realworld_baseline_{config['batch_size']}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={**config},
    )

    train_results_list = []
    test_results_list = []
    start_time = time.time()

    log_date = time.strftime("%Y%m%d")
    log_timestamp = time.strftime("%H%M%S")
    if config["model"] == "DeepConvLSTM":
        model = DeepConvLSTM(
            n_channels=dataset_train.n_channels,
            n_classes=dataset_train.n_classes,
            dataset="realworld",
            experiment=f"/{log_date}/{log_timestamp}",
        ).cuda()
    else:
        model = AttendAndDiscriminate(
            input_dim=dataset_train.n_channels,
            num_class=dataset_train.n_classes,
            dataset="realworld",
            experiment=f"/{log_date}/{log_timestamp}",
        ).cuda()

    sys.stdout = Logger(os.path.join(model.path_logs, "log"))

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
        ) = train_model(
            model,
            dataset_train,
            dataset_val,
            dataset_test,
            lr=config["lr"],
            class_weights=class_weights,
            seed=seed,
            n_epochs=config["epochs"],
            batch_size_train=config["batch_size"],
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

    # run_train_analysis(train_results)
    run_test_analysis(test_results)

    wandb_logging(train_results, test_results, {**config})
    wandb.finish()
