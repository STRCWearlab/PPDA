import os
import sys
import time
from datetime import timedelta
import random
import argparse

sys.path.append("..")

import pandas as pd
import wandb

import numpy as np
from sklearn import metrics
from sklearn.utils import class_weight
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader

from simuaug.utils import get_default_device, paint, seed_torch
from simuaug.datasets import SensorDataset

from models import DeepConvLSTM, AttendAndDiscriminate
from models.utils import init_weights, wandb_logging, AverageMeter, Logger
from models.utils import run_train_analysis, run_test_analysis


# Train one epoch
def train_one_epoch(model, loader, criterion, optimizer, verbose=True, print_freq=100):
    losses = AverageMeter("Loss")
    model.train()
    for batch_idx, (data, target, idx) in enumerate(loader):
        data = data.cuda()
        target = target.view(-1).cuda()
        z, logits = model(data)
        loss = criterion(logits, target)
        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            if batch_idx % print_freq == 0:
                print(f"[-] Batch {batch_idx + 1}/{len(loader)}\t Loss: {str(losses)}")


def eval_model(model, eval_data, criterion=None, batch_size=256, seed=1):
    """
    Evaluate trained model.

    :param model: A trained model which is to be evaluated.
    :param eval_data: A SensorDataset containing the data to be used for evaluating the model.
    :param criterion: Citerion object which was used during training of model.
    :param batch_size: Batch size to use during evaluation.
    :param seed: Random seed which is employed.

    :return: loss, accuracy, f1 weighted and macro for evaluation data; if return_results, also predictions
    """

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    print(paint("Running HAR evaluation loop ..."))

    loader_test = DataLoader(
        eval_data,
        batch_size,
        False,
        pin_memory=False,
        worker_init_fn=np.random.seed(int(seed)),
    )

    print("[-] Loading checkpoint ...")

    path_checkpoint = os.path.join(model.path_checkpoints, "best.pth")

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    start_time = time.time()
    loss_test, acc_test, fm_test, fw_test = eval_one_epoch(
        model, loader_test, criterion
    )

    print(
        paint(
            f"[-] Test loss: {loss_test:.2f}"
            f"\tacc: {100 * acc_test:.2f}(%)\tfm: {100 * fm_test:.2f}(%)\tfw: {100 * fw_test:.2f}(%)"
        )
    )

    elapsed = round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    print(paint(f"Finished HAR evaluation loop (h:m:s): {elapsed}"))

    return loss_test, acc_test, fm_test, fw_test, elapsed


def eval_one_epoch(model, loader, criterion):
    """
    Eval model for a one of epoch.

    :param model: A trained model which is to be evaluated.
    :param loader: A DataLoader object containing the data to be used for evaluating the model.
    :param criterion: The loss object.
    :return: loss, accuracy, f1 weighted and macro for evaluation data
    """

    losses = AverageMeter("Loss")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, idx) in enumerate(loader):
            data.cuda()
            target.cuda()

            z, logits = model(data)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.shape[0])
            probabilities = torch.nn.Softmax(dim=1)(logits)
            _, predictions = torch.max(probabilities, 1)

            y_pred.append(predictions.cpu().numpy().reshape(-1))
            y_true.append(target.cpu().numpy().reshape(-1))

    # append invalid samples at the beginning of the test sequence
    # if loader.dataset.prefix == "test":
    #     ws = data.shape[1] - 1
    #     samples_invalid = [y_true[0][0]] * ws
    #     y_true.append(samples_invalid)
    #     y_pred.append(samples_invalid)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    acc = metrics.accuracy_score(y_true, y_pred)
    fm = metrics.f1_score(y_true, y_pred, average="macro")
    fw = metrics.f1_score(y_true, y_pred, average="weighted")

    return losses.avg, acc, fm, fw


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

    # Store the training and validation metrics
    t_loss, t_acc, t_fm, t_fw = [], [], [], []
    v_loss, v_acc, v_fm, v_fw = [], [], [], []

    for epoch in range(n_epochs):
        if verbose:
            print(
                f"------------------------------ Epoch {epoch + 1}/{n_epochs} ------------------------------\n"
                f"Learning rate: {optimizer.param_groups[0]['lr']}"
            )
        train_one_epoch(model, loader_train, criterion, optimizer)

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


def get_config():
    parser = argparse.ArgumentParser(description="Configuration for experiment")

    # Adding all the parameters from your config
    parser.add_argument(
        "--model", type=str, default="DeepConvLSTM", help="DeepLearning model"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--use_weights", action="store_true", help="Use weighted loss for training"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3], help="Random seeds"
    )
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument("--window", type=int, default=60, help="Batch size")
    parser.add_argument("--stride", type=int, default=30, help="Batch size")
    args = parser.parse_args()
    config = {
        "model": args.model,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "use_weights": args.use_weights,
        "offline": args.offline,
        "window": args.window,
        "stride": args.stride,
    }

    return config


if __name__ == "__main__":
    sys.path.append("..")
    config = get_config()
    load_dotenv()

    config = {
        "model": config["model"],
        "window": config["window"],
        "stride": config["stride"],
        "dataset_name": "mmfit",
        "train-prefix": "train",
        "val-prefix": "val",
        "test-prefix": "test",
        "path_processed": "../data/mmfit/",
        "lazy_load": True,
        "use_weights": config["use_weights"],
        "batch_size": config["batch_size"],
        "lr": 0.001,
        "lr_step": 10,
        "lr_decay": 0.9,
        "weights_init": "orthogonal",
        "seeds": config["seeds"],
        "epochs": config["epochs"],
        "verbose": True,
        "offline": config["offline"],
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

    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "SPECIFY_YOUR_WANDB_ENTITY")
    WANDB_PROJECT = "simuaug_mmfit"
    WANDB_RUN_NAME = f"mmfit_baseline_{config['batch_size']}"

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        mode="offline" if config["offline"] else "online",
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
            class_weights=class_weights,
            seed=seed,
            n_epochs=config["epochs"],
            batch_size_train=config["batch_size"],
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
