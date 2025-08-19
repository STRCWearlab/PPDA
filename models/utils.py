import os
import sys
import errno
import json

import pandas as pd
import numpy as np

import wandb
import torch
from torch import nn

from simuaug.utils import paint


def makedir(path):
    """
    Creates a directory if not already exists.

    :param str path: The path which is to be created.
    :return: None
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.exists:
        print(f"[+] Created directory in {path}")


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name, fmt=":4f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            makedir(os.path.dirname(fpath))
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def init_weights(model, method):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network)

    :param model: network of which weights are to be initialised
    :param str method: Method to initialise weights
    :return: network with initialised weights
    """
    for m in model.modules():
        if isinstance(m, nn.Linear) or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            if method == "normal":
                torch.nn.init.normal_(m.weight)
            elif method == "orthogonal":
                torch.nn.init.orthogonal_(m.weight)
            elif method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            elif method == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            elif method == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            elif method == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
            # LSTM initialisation
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    if method == "normal":
                        torch.nn.init.normal_(param.data)
                    elif method == "orthogonal":
                        torch.nn.init.orthogonal_(param.data)
                    elif method == "xavier_uniform":
                        torch.nn.init.xavier_uniform_(param.data)
                    elif method == "xavier_normal":
                        torch.nn.init.xavier_normal_(param.data)
                    elif method == "kaiming_uniform":
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif method == "kaiming_normal":
                        torch.nn.init.kaiming_normal_(param.data)
                elif "weight_hh" in name:
                    if method == "normal":
                        torch.nn.init.normal_(param.data)
                    elif method == "orthogonal":
                        torch.nn.init.orthogonal_(param.data)
                    elif method == "xavier_uniform":
                        torch.nn.init.xavier_uniform_(param.data)
                    elif method == "xavier_normal":
                        torch.nn.init.xavier_normal_(param.data)
                    elif method == "kaiming_uniform":
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif method == "kaiming_normal":
                        torch.nn.init.kaiming_normal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0.0)
    return model


def run_train_analysis(train_results):
    """
    Runs an average and subject-wise analysis of saved train results.

    :param train_results: the train result dataframe returned by the cross_validate function.
    :return: None
    """
    # average analysis
    avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw = [], [], [], []
    avg_v_loss, avg_v_acc, avg_v_fm, avg_v_fw = [], [], [], []

    # average analysis
    print(paint("AVERAGE RESULTS"))
    for i, row in train_results.iterrows():
        if i == 0:
            avg_t_loss = np.asarray(row["t_loss"])
            avg_t_acc = np.asarray(row["t_acc"])
            avg_t_fm = np.asarray(row["t_fm"])
            avg_t_fw = np.asarray(row["t_fw"])
            avg_v_loss = np.asarray(row["v_loss"])
            avg_v_acc = np.asarray(row["v_acc"])
            avg_v_fm = np.asarray(row["v_fm"])
            avg_v_fw = np.asarray(row["v_fw"])
        else:
            avg_t_loss = np.add(avg_t_loss, row["t_loss"])
            avg_t_acc = np.add(avg_t_acc, row["t_acc"])
            avg_t_fm = np.add(avg_t_fm, row["t_fm"])
            avg_t_fw = np.add(avg_t_fw, row["t_fw"])
            avg_v_loss = np.add(avg_v_loss, row["v_loss"])
            avg_v_acc = np.add(avg_v_acc, row["v_acc"])
            avg_v_fm = np.add(avg_v_fm, row["v_fm"])
            avg_v_fw = np.add(avg_v_fw, row["v_fw"])

    avg_t_loss /= len(train_results)
    avg_t_acc /= len(train_results)
    avg_t_fm /= len(train_results)
    avg_t_fw /= len(train_results)
    avg_v_loss /= len(train_results)
    avg_v_acc /= len(train_results)
    avg_v_fm /= len(train_results)
    avg_v_fw /= len(train_results)

    print("\nAverage Train results (last epoch):")
    print(
        "Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}".format(
            avg_t_loss[-1], avg_t_acc[-1], avg_t_fm[-1], avg_t_fw[-1]
        )
    )
    print("\nAverage Validation results (last epoch):")
    print(
        "Loss: {:.4f} - Accuracy: {:.4f} - F1-score (macro): {:.4f} - F1-score (weighted): {:.4f}".format(
            avg_v_loss[-1], avg_v_acc[-1], avg_v_fm[-1], avg_v_fw[-1]
        )
    )


def run_test_analysis(test_results):
    """
    Runs an average analysis of saved test results.

    :param test_results: the test result dataframe returned by the cross_validate function.
    :return: None
    """
    if test_results is not None:
        avg_t_loss, avg_t_acc, avg_t_fm, avg_t_fw = 0.0, 0.0, 0.0, 0.0
        t_loss_list, t_acc_list, t_fm_list, t_fw_list = [], [], [], []
        # average analysis
        for i, row in test_results.iterrows():
            if i == 0:
                avg_t_loss = np.asarray(row["test_loss"])
                avg_t_acc = np.asarray(row["test_acc"])
                avg_t_fm = np.asarray(row["test_fm"])
                avg_t_fw = np.asarray(row["test_fw"])

                t_loss_list.append(row["test_loss"])
                t_acc_list.append(row["test_acc"])
                t_fm_list.append(row["test_fm"])
                t_fw_list.append(row["test_fw"])
            else:
                avg_t_loss = np.add(avg_t_loss, row["test_loss"])
                avg_t_acc = np.add(avg_t_acc, row["test_acc"])
                avg_t_fm = np.add(avg_t_fm, row["test_fm"])
                avg_t_fw = np.add(avg_t_fw, row["test_fw"])

                t_loss_list.append(row["test_loss"])
                t_acc_list.append(row["test_acc"])
                t_fm_list.append(row["test_fm"])
                t_fw_list.append(row["test_fw"])

        if len(t_loss_list) > 0:
            print(
                f"Loss: {np.mean(t_loss_list):.4f}\u00B1{np.std(t_loss_list):.4f}",
                end="",
            )
            print(
                f"- Accuracy: {np.mean(t_acc_list):.4f}\u00B1{np.std(t_acc_list):.4f}",
                end="",
            )
            print(
                f"- F1-score (macro): {np.mean(t_fm_list):.4f}\u00B1{np.std(t_fm_list):.4f}",
                end="",
            )
            print(
                f"- F1-score: {np.mean(t_fw_list):.4f}\u00B1{np.std(t_fw_list):.4f}",
                end="",
            )


def rerun_analysis(log_directory):
    """
    Method used to rerun an analysis by loading up saved train and (if applicable) test results.

    :param log_directory: directory where results were saved to (e.g. 20211205/225740)
    :return: None
    """
    train_results_df = pd.read_csv(
        os.path.join("../logs", log_directory, "train_results.csv"), index_col=None
    )
    train_results_df[
        ["t_loss", "t_acc", "t_fm", "t_fw", "v_loss", "v_acc", "v_fm", "v_fw"]
    ] = train_results_df[
        ["t_loss", "t_acc", "t_fm", "t_fw", "v_loss", "v_acc", "v_fm", "v_fw"]
    ].apply(
        lambda x: list(map(json.loads, x))
    )
    run_train_analysis(train_results_df)
    if os.path.isfile(os.path.join("../logs", log_directory, "test_results.csv")):
        test_results_df = pd.read_csv(
            os.path.join("../logs", log_directory, "test_results.csv"), index_col=None
        )
        run_test_analysis(test_results_df)


def wandb_logging(train_results, test_results, config):
    t_loss, t_acc, t_fw, t_fm = (
        np.zeros(config["epochs"]),
        np.zeros(config["epochs"]),
        np.zeros(config["epochs"]),
        np.zeros(config["epochs"]),
    )
    v_loss, v_acc, v_fw, v_fm = (
        np.zeros(config["epochs"]),
        np.zeros(config["epochs"]),
        np.zeros(config["epochs"]),
        np.zeros(config["epochs"]),
    )

    for i in range(len(train_results)):
        t_loss = np.add(t_loss, train_results["t_loss"][i])
        t_acc = np.add(t_acc, train_results["t_acc"][i])
        t_fw = np.add(t_fw, train_results["t_fw"][i])
        t_fm = np.add(t_fm, train_results["t_fm"][i])

        v_loss = np.add(v_loss, train_results["v_loss"][i])
        v_acc = np.add(v_acc, train_results["v_acc"][i])
        v_fw = np.add(v_fw, train_results["v_fw"][i])
        v_fm = np.add(v_fm, train_results["v_fm"][i])

    table = wandb.Table(
        data=[
            [a, b, c, d, e, f, g, h, i]
            for (a, b, c, d, e, f, g, h, i) in zip(
                list(range(config["epochs"])),
                t_loss / len(train_results),
                t_acc / len(train_results),
                t_fw / len(train_results),
                t_fm / len(train_results),
                v_loss / len(train_results),
                v_acc / len(train_results),
                v_fw / len(train_results),
                v_fm / len(train_results),
            )
        ],
        columns=[
            "epochs",
            "t_loss",
            "t_acc",
            "t_fw",
            "t_fm",
            "v_loss",
            "v_acc",
            "v_fw",
            "v_fm",
        ],
    )

    wandb.log(
        {
            "train_loss": wandb.plot.line(
                table, "epochs", "t_loss", title="Train Loss"
            ),
            "train_acc": wandb.plot.line(
                table, "epochs", "t_acc", title="Train Accuracy"
            ),
            "train_fm": wandb.plot.line(
                table, "epochs", "t_fm", title="Train F1-macro"
            ),
            "train_fw": wandb.plot.line(
                table, "epochs", "t_fw", title="Train F1-weighted"
            ),
            "val_loss": wandb.plot.line(table, "epochs", "v_loss", title="Valid Loss"),
            "val_acc": wandb.plot.line(
                table, "epochs", "v_acc", title="Valid Accuracy"
            ),
            "val_fm": wandb.plot.line(table, "epochs", "v_fm", title="Valid F1-macro"),
            "val_fw": wandb.plot.line(
                table, "epochs", "v_fw", title="Valid F1-weigthed"
            ),
        }
    )

    if test_results is not None:
        wandb.log(
            {
                "test_loss": test_results["test_loss"].mean(),
                "test_acc": test_results["test_acc"].mean(),
                "test_fm": test_results["test_fm"].mean(),
                "test_fw": test_results["test_fw"].mean(),
            }
        )


# Match the size of the data loaders
def cycle_loader(loader):
    # itertools.cycle() cannot be used here as it will not run collate_fn after the 1st iteration
    while True:
        for batch in loader:
            # Each iteration will apply collate_fn
            yield batch


def calc_epsilon(grads, lower_bound=1e-10, upper_bound=1e-2):
    grad_norm = torch.sqrt(sum(torch.norm(grad) ** 2 for grad in grads))
    epsilon = 0.01 / grad_norm
    epsilon = torch.clamp(epsilon, min=lower_bound, max=upper_bound)
    return epsilon


def set_model_params(model, params):
    """
    Set model parameters to the given values

    :param model:
    :param params:
    :return:
    """
    with torch.no_grad():
        for param, new_param in zip(model.parameters(), params):
            param.copy_(new_param)
