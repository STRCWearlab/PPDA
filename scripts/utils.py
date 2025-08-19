import os

import wandb
import numpy as np
import time
from datetime import timedelta
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from models.utils import AverageMeter, cycle_loader, calc_epsilon, set_model_params
from simuaug.utils import get_default_device, paint, seed_torch


def wandb_logging(train_results, test_results, config):
    """
    Logs final results and averages to WandB.

    :param train_results: DataFrame of training results.
    :param test_results: DataFrame of test results.
    :param config: Dictionary containing configuration parameters.
    """
    # Extract the final metrics for training results
    train_results["final_t_loss"] = train_results["t_loss"].apply(
        lambda x: x[-1] if isinstance(x, list) else x
    )
    train_results["final_t_acc"] = train_results["t_acc"].apply(
        lambda x: x[-1] if isinstance(x, list) else x
    )
    train_results["final_t_fm"] = train_results["t_fm"].apply(
        lambda x: x[-1] if isinstance(x, list) else x
    )
    train_results["final_t_fw"] = train_results["t_fw"].apply(
        lambda x: x[-1] if isinstance(x, list) else x
    )

    train_results["final_v_loss"] = train_results["v_loss"].apply(
        lambda x: x[-1] if isinstance(x, list) else x
    )
    train_results["final_v_acc"] = train_results["v_acc"].apply(
        lambda x: x[-1] if isinstance(x, list) else x
    )
    train_results["final_v_fm"] = train_results["v_fm"].apply(
        lambda x: x[-1] if isinstance(x, list) else x
    )
    train_results["final_v_fw"] = train_results["v_fw"].apply(
        lambda x: x[-1] if isinstance(x, list) else x
    )

    # Average and standard deviation for training results
    avg_t_loss = train_results["final_t_loss"].mean()
    std_t_loss = train_results["final_t_loss"].std()

    avg_t_acc = train_results["final_t_acc"].mean()
    std_t_acc = train_results["final_t_acc"].std()

    avg_t_fm = train_results["final_t_fm"].mean()
    std_t_fm = train_results["final_t_fm"].std()

    avg_t_fw = train_results["final_t_fw"].mean()
    std_t_fw = train_results["final_t_fw"].std()

    # Average and standard deviation for validation results
    avg_v_loss = train_results["final_v_loss"].mean()
    std_v_loss = train_results["final_v_loss"].std()

    avg_v_acc = train_results["final_v_acc"].mean()
    std_v_acc = train_results["final_v_acc"].std()

    avg_v_fm = train_results["final_v_fm"].mean()
    std_v_fm = train_results["final_v_fm"].std()

    avg_v_fw = train_results["final_v_fw"].mean()
    std_v_fw = train_results["final_v_fw"].std()

    # Log training and validation results
    wandb.log(
        {
            "train_loss": f"{avg_t_loss:.4f} ± {std_t_loss:.4f}",
            "train_acc": f"{avg_t_acc:.4f} ± {std_t_acc:.4f}",
            "train_fm": f"{avg_t_fm:.4f} ± {std_t_fm:.4f}",
            "train_fw": f"{avg_t_fw:.4f} ± {std_t_fw:.4f}",
            "val_loss": f"{avg_v_loss:.4f} ± {std_v_loss:.4f}",
            "val_acc": f"{avg_v_acc:.4f} ± {std_v_acc:.4f}",
            "val_fm": f"{avg_v_fm:.4f} ± {std_v_fm:.4f}",
            "val_fw": f"{avg_v_fw:.4f} ± {std_v_fw:.4f}",
        }
    )

    # Log test results, if available
    if test_results is not None:
        avg_test_loss = test_results["test_loss"].mean()
        avg_test_acc = test_results["test_acc"].mean()
        avg_test_fm = test_results["test_fm"].mean()
        avg_test_fw = test_results["test_fw"].mean()

        # Average and standard deviation for validation results
        std_test_loss = test_results["test_loss"].std()
        std_test_acc = test_results["test_acc"].std()
        std_test_fm = test_results["test_fm"].std()
        std_test_fw = test_results["test_fw"].std()

        wandb.log(
            {
                "test_loss": avg_test_loss,
                "test_acc": avg_test_acc,
                "test_fm": avg_test_fm,
                "test_fw": avg_test_fw,
                "_test_loss": f"{avg_test_loss:.4f} ± {std_test_loss:.4f}",
                "_test_acc": f"{avg_test_acc:.4f} ± {std_test_acc:.4f}",
                "_test_fm": f"{avg_test_fm:.4f} ± {std_test_fm:.4f}",
                "_test_fw": f"{avg_test_fw:.4f} ± {std_test_fw:.4f}",
            }
        )


def run_test_analysis(test_results):
    """
    Runs an average analysis of final test results.

    :param test_results: the test result dataframe returned by the cross_validate function.
    :return: None
    """
    if test_results is not None:
        # Initialize accumulators for metrics
        t_loss_list, t_acc_list, t_fm_list, t_fw_list = [], [], [], []

        # Collect the final test results
        for _, row in test_results.iterrows():
            t_loss_list.append(row["test_loss"])
            t_acc_list.append(row["test_acc"])
            t_fm_list.append(row["test_fm"])
            t_fw_list.append(row["test_fw"])

        # Compute mean and standard deviation for each metric
        avg_loss = np.mean(t_loss_list)
        std_loss = np.std(t_loss_list)

        avg_acc = np.mean(t_acc_list)
        std_acc = np.std(t_acc_list)

        avg_fm = np.mean(t_fm_list)
        std_fm = np.std(t_fm_list)

        avg_fw = np.mean(t_fw_list)
        std_fw = np.std(t_fw_list)

        # Print the summary
        print(
            f"Loss: {avg_loss:.4f} ± {std_loss:.4f} "
            f"- Accuracy: {avg_acc:.4f} ± {std_acc:.4f} "
            f"- F1-score (macro): {avg_fm:.4f} ± {std_fm:.4f} "
            f"- F1-score (weighted): {avg_fw:.4f} ± {std_fw:.4f}"
        )


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


def train_one_epoch(model, loader, criterion, optimizer, verbose=True, print_freq=100):
    """
    Train the model for one epoch. Updated to include logging of augmentation statistics.

    :param model:
    :param loader:
    :param criterion:
    :param optimizer:
    :param verbose:
    :param print_freq:
    :return:
    """
    losses = AverageMeter("Loss")
    model.train()

    log_aug_stats = False
    if loader.aug_planner is not None:
        log_aug_stats = True
        log_aug_list = []

    for batch_idx, (data, target, idx) in enumerate(loader):
        data = data.cuda()
        target = target.view(-1).cuda()
        z, logits = model(data)
        loss = criterion(logits, target)
        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_aug_stats:
            log_aug_list.append(
                loader.aug_planner.current_subpolicy_one_hot.detach().cpu().numpy()
            )

        if verbose:
            if batch_idx % print_freq == 0:
                print(f"[-] Batch {batch_idx + 1}/{len(loader)}\t Loss: {str(losses)}")

        # If logging augmentation stats, calculate the total and fraction
    if log_aug_stats:
        # Convert log_aug_list to a numpy array for easier summation
        log_aug_array = np.array(log_aug_list)

        # Sum across all batches to get absolute numbers for each augmentation
        total_aug_counts = np.sum(log_aug_array, axis=0)

        # Calculate the fractions (percentages) of each augmentation applied
        total_batches = len(loader)
        aug_fractions = total_aug_counts / total_batches

        # Print the absolute counts and fractions
        aug_fractions_str = ", ".join([f"{x:.4f}" for x in aug_fractions])
        print(
            f"[-] Aug stats: counts: {total_aug_counts}, ratios: [{aug_fractions_str}]"
        )



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

    if len(p_subpolicies) > 100:
        p_subpolicies_np = p_subpolicies.detach().cpu().numpy()
        total_aug_counts_np = np.array(total_aug_counts)
        aug_fractions_np = np.array(aug_fractions)

        # Sort subpolicies by probability
        sorted_indices = np.argsort(-p_subpolicies_np)  # Descending order
        top_indices = sorted_indices[:10]  # Get top 10 indices
        print(f"[-] Displaying top 10 subpolicies out of {len(p_subpolicies_np)}:")
        for idx in top_indices:
            print(
                f"  Subpolicy {idx:3d}: Probability = {p_subpolicies_np[idx]:.4f}, "
                f"Count = {total_aug_counts_np[idx]}, Ratio = {aug_fractions_np[idx]:.4f}"
            )

    else:
        print_augmentation_statistics(aug_planner)
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


def print_augmentation_statistics(aug_planner):
    a_subpolicies_sigmoid = torch.sigmoid(aug_planner.a_subpolicies)

    # Normalize to ensure the probabilities sum to 1
    p_subpolicies = a_subpolicies_sigmoid / a_subpolicies_sigmoid.sum()
    p_subpolicies_formatted = ", ".join(
        [f"{prob:.4f}" for prob in p_subpolicies.detach().cpu().numpy()]
    )
    # Print the absolute counts and fractions
    print(paint(f"[-] Aug stats: p_subpolicies: {p_subpolicies_formatted}"))


# Train one epoch
def train_one_epoch_baseline(model, loader, criterion, optimizer, verbose=True, print_freq=100):
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
