import torch
from itertools import chain
from torch.utils.data import Sampler

import random
import copy
from torch.utils.data import DataLoader
from collections import defaultdict

from simuaug.augmentations import stda, ppda
from simuaug.datasets import SensorDataset

import wimusim
from wimusim.datasets import WIMUSimDataset
from simuaug.augmentations import AugPolicyPlanner
import numpy as np


## For n_sub_baseline evaluations
class FixedLengthDataLoader(DataLoader):
    def __init__(self, dataset, fixed_length, *args, **kwargs):
        super(FixedLengthDataLoader, self).__init__(dataset, *args, **kwargs)
        self.fixed_length = fixed_length

    def __len__(self):
        return self.fixed_length  # Use specified number of batches

    def __iter__(self):
        base_iter = super(FixedLengthDataLoader, self).__iter__()
        for _ in range(self.fixed_length):
            try:
                X, y, idx = next(base_iter)
            except StopIteration:
                # Restart the base iterator when data is exhausted
                base_iter = super(FixedLengthDataLoader, self).__iter__()
                X, y, idx = next(base_iter)
            yield X, y, idx


class STDADataLoader(DataLoader):
    def __init__(
        self,
        dataset: SensorDataset,
        aug_planner,
        sample,
        eval=False,
        n_batches_per_epoch=None,
        *args,
        **kwargs,
    ):
        super(STDADataLoader, self).__init__(dataset, *args, **kwargs)
        self.aug_planner = aug_planner
        self.sample = sample
        self.eval = eval
        self.n_batches_per_epoch = n_batches_per_epoch

    def __len__(self):
        if self.n_batches_per_epoch is not None:
            return self.n_batches_per_epoch  # Use specified number of batches
        else:
            return super().__len__()

    def __iter__(self):
        base_iter = super(STDADataLoader, self).__iter__()
        batch_count = 0
        while True:
            try:
                # Get the next batch from the base iterator
                X, y, idx = next(base_iter)
            except StopIteration:
                if self.n_batches_per_epoch is None:
                    break
                # Restart the base iterator when data is exhausted
                base_iter = super(STDADataLoader, self).__iter__()
                X, y, idx = next(base_iter)

            if not self.eval:
                # Sample sub-policy based on a_subpolicies unless one_hot is given
                # For training dataset, we want to sample a new sub-policy for each batch
                # For validation dataset, we want to use the same sub-policy as the training batch
                if self.sample:
                    self.aug_planner.sample_subpolicy()

                # Apply the augmentations from the current subpolicy
                for augmentor in self.aug_planner.current_subpolicy:
                    if isinstance(augmentor, stda.TimeScaling):
                        # Sample a scaling factor and re-fetch data with the new scale
                        scale_factor: float = augmentor.sample_scaling_factor()
                        T_orig = X.shape[1]  # T is the time dimension (N, T, C)

                        if scale_factor > 1.0:
                            # Re-fetch data with the scaled window size
                            X = torch.stack(
                                [
                                    self.dataset.__getitem__(i, scale=scale_factor)[0]
                                    for i in idx
                                ]
                            )

                        # Apply time scaling if needed
                        X = augmentor.apply_augmentation(X=X, T_orig=T_orig)

                    else:
                        X = augmentor.apply_augmentation(X=X)

                # Multiply X by the one-hot vector for proper gradient flow
                # This ensures one-hot_vector is still involved in the computation
                # This won't change the value of X
                for e in self.aug_planner.current_subpolicy_one_hot:
                    if e == 1.0:
                        X = e * X

                yield X, y, idx
                batch_count += 1

            else:
                # In evaluation mode, we don't want to apply any augmentation
                # And just pass the sampled policy
                # print("Eval mode. Only passing the sampled policy.")
                for e in self.aug_planner.current_subpolicy_one_hot:
                    if e == 1.0:
                        X = e * X

                yield X, y, idx
                batch_count += 1

            if (
                self.n_batches_per_epoch is not None
                and batch_count >= self.n_batches_per_epoch
            ):
                break


class PPDADataLoader(DataLoader):
    dataset: WIMUSimDataset

    def __init__(
        self,
        dataset: WIMUSimDataset,
        aug_planner,
        sample: bool = True,
        n_batches_per_epoch=None,
        *args,
        **kwargs,
    ):
        super(PPDADataLoader, self).__init__(dataset, *args, **kwargs)
        self.aug_planner: AugPolicyPlanner = aug_planner
        self.sample = sample  # Whether to sample a new sub-policy for each batch
        self.wimusim_env = None
        self.device = dataset.device
        self.n_batches_per_epoch = n_batches_per_epoch

        self._init_wimusim_env()

    def __len__(self):
        if self.n_batches_per_epoch is not None:
            return self.n_batches_per_epoch  # Use specified number of batches
        else:
            return super().__len__()

    def _init_wimusim_env(self):
        self.wimusim_env = wimusim.WIMUSim(
            B=self.dataset.B_list[0],
            D=self.dataset.D_list[0],
            P=self.dataset.P_list[0],
            H=self.dataset.H_list[0],
            device=self.dataset.device,
        )

    def __iter__(self):
        # Create an iterator for the base DataLoader
        base_iter = super(PPDADataLoader, self).__iter__()
        batch_count = 0

        while True:
            try:
                # Get the next batch from the base iterator
                X_d, y, (idx, list_ids) = next(base_iter)
            except StopIteration:
                if self.n_batches_per_epoch is None:
                    break
                # Restart the base iterator when data is exhausted
                base_iter = super(PPDADataLoader, self).__iter__()
                X_d, y, (idx, list_ids) = next(base_iter)

            # list_ids specifies which P, B, H to use for the current batch

            # Sample sub-policy based on a_subpolicies unless one_hot is given
            # For training dataset, we want to sample a new sub-policy for each batch
            # For validation dataset, we want to use the same sub-policy as the training batch
            if self.sample:
                self.aug_planner.sample_subpolicy()

            virtual_IMU_dict = {
                imu_name: {"acc": [], "gyro": []}
                for imu_name in self.dataset.P_list[0].imu_names
            }
            y_list = []
            idx_list = []
            list_id_list = []
            # We can use the same TimeScaling
            for augmentor in self.aug_planner.current_subpolicy:
                # TimeScaling should be applied to the data.
                if isinstance(augmentor, ppda.TimeScaling):
                    # Sample a scaling factor and re-fetch data with the new scale
                    scale_factor: float = augmentor.sample_scaling_factor()
                    T_orig = X_d.shape[2]  # T is the time dimension (N, J, T, C)
                    if scale_factor > 1.0:
                        # TODO: Make this faster.
                        idx = torch.clamp(idx, 0, X_d.shape[0] - 1)
                        # Re-fetch data with the scaled window size
                        # Loading the entire dataset is faster than loading one by one with randomized indices
                        # X_d = torch.stack(
                        #     [
                        #         wimusim_dataset.__getitem__(i, scale=1.1)[0]
                        #         for i in idx
                        #     ]
                        # )
                        X_d = torch.stack(
                            [
                                self.dataset.__getitem__(i, scale=scale_factor)[0]
                                for i in range(self.dataset.__len__())
                            ]
                        )[idx]
                    # Apply time scaling if needed
                    X_d = augmentor.apply_augmentation(X=X_d, T_orig=T_orig)
                elif isinstance(augmentor, ppda.TimeWarping):
                    X_d = augmentor.apply_augmentation(X=X_d)
                elif isinstance(augmentor, ppda.MagnitudeWarping):
                    X_d = augmentor.apply_augmentation(X=X_d)

            # Data Augmentation involving B, P, H are applied here
            for list_id in torch.unique(list_ids):
                # print(f"processing list_id {list_id}...")
                X_d_sbj = X_d[list_ids == list_id]  # Get the data of the subject.
                y_sbj = y[list_ids == list_id]  # Get the target of the subject.
                idx_sbj = idx[list_ids == list_id]  # Get the index of the subject.

                B = copy.deepcopy(self.dataset.B_list[list_id])
                P = copy.deepcopy(self.dataset.P_list[list_id])
                H = copy.deepcopy(
                    self.dataset.H_list[list_id]
                )  # TODO: We don't really need to use this as it will be overwritten by the WIMUSim object

                # We can use the same TimeScaling
                for augmentor in self.aug_planner.current_subpolicy:
                    if isinstance(augmentor, ppda.Rotation):
                        if augmentor.paramix:
                            # Select P randomly from the P_list if paramix is enabled
                            P = copy.deepcopy(random.choice(self.dataset.P_list))
                        P = augmentor.apply_augmentation(P=P)
                    elif isinstance(augmentor, ppda.NoiseBias):
                        if hasattr(H, "imu_names"):
                            H = augmentor.apply_augmentation(H=H)
                        else:
                            # Use P.imu_names if H does not have imu_names
                            H = augmentor.apply_augmentation(H=H, imu_names=P.imu_names)

                # Transform D_data to WIMUSim.Dynamics
                D = wimusim.WIMUSim.Dynamics(
                    translation={"XYZ": X_d_sbj[:, 0, :, 1:].detach().cpu().numpy()},
                    orientation={
                        joint_name: X_d_sbj[:, joint_id, :, :].detach().cpu().numpy()
                        for joint_name, joint_id in self.dataset._D_ori_key_idx.items()
                    },
                    sample_rate=self.wimusim_env.D.sample_rate,
                    device=self.device,
                )

                # Apply the augmentations from the current subpolicy
                # for augmentor in self.aug_planner.current_subpolicy:
                for augmentor in self.aug_planner.current_subpolicy:
                    if isinstance(augmentor, ppda.MagnitudeScaling):
                        D = augmentor.apply_augmentation(D=D)

                if self.wimusim_env is None:
                    self.wimusim_env = wimusim.WIMUSim(B, D, P, H, device=self.device)
                else:
                    # Reuse the WIMUSim object if it already exists
                    self.wimusim_env.B = B
                    self.wimusim_env.D = D
                    self.wimusim_env.P = P
                    self.wimusim_env.H = H

                virtual_IMU_dict_sbj = self.wimusim_env.simulate(mode="generate")
                for imu_name, imu_data in virtual_IMU_dict_sbj.items():
                    virtual_IMU_dict[imu_name]["acc"].append(imu_data[0])
                    virtual_IMU_dict[imu_name]["gyro"].append(imu_data[1])
                y_list.append(y_sbj)
                idx_list.append(idx_sbj)
                list_id_list.append(list_ids[list_ids == list_id])

            if self.dataset.acc_only:
                X_res = torch.concat(
                    [
                        torch.concat(virtual_IMU_dict[imu_name]["acc"], dim=0)
                        for imu_name in self.dataset.P_list[0].imu_names
                    ],
                    dim=-1,
                )
            elif self.dataset.gyro_only:
                X_res = torch.concat(
                    [
                        torch.concat(virtual_IMU_dict[imu_name]["gyro"], dim=0)
                        for imu_name in self.dataset.P_list[0].imu_names
                    ],
                    dim=-1,
                )
            else:
                if self.dataset.data_order == "alternate":
                    X_res = torch.concat(
                        [
                            torch.concat(virtual_IMU_dict[imu_name][sensor_type], dim=0)
                            for imu_name in self.dataset.P_list[0].imu_names
                            for sensor_type in ["acc", "gyro"]
                        ],
                        dim=-1,
                    )
                elif self.dataset.data_order == "sequential":
                    X_res = torch.concat(
                        [
                            torch.concat(virtual_IMU_dict[imu_name][sensor_type], dim=0)
                            for sensor_type in ["acc", "gyro"]
                            for imu_name in self.dataset.P_list[0].imu_names
                        ],
                        dim=-1,
                    )
                else:
                    raise ValueError("Invalid order", self.dataset.data_order)

            y = torch.concat(y_list, dim=0).to(self.device)
            idx = torch.concat(idx_list, dim=0)
            list_ids = torch.concat(list_id_list, dim=0)

            # Apply scaling (standardization)
            X = (
                (
                    (X_res.detach().cpu() - self.dataset.scale_config["mean"])
                    / self.dataset.scale_config["std"]
                )
                .to(self.device)
                .float()
            )

            # Multiply X by the one-hot vector for proper gradient flow
            # This ensures one-hot_vector is still involved in the computation
            # This won't change the value of X
            for e in self.aug_planner.current_subpolicy_one_hot:
                if e == 1.0:
                    X = e * X

            yield X, y, (idx, list_ids)
            batch_count += 1

            if (
                self.n_batches_per_epoch is not None
                and batch_count >= self.n_batches_per_epoch
            ):
                break


class RealWorldCustomSampler(Sampler):
    def __init__(self, dataset, batch_size, max_list_ids_per_batch=13):
        """
        :param dataset: The dataset to sample from
        :param batch_size: Number of samples in each batch
        :param max_list_ids_per_batch: Maximum number of unique list_ids in each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_list_ids_per_batch = max_list_ids_per_batch

        # Use a dictionary to map class labels to list_ids
        self.class_to_list_ids = defaultdict(list)
        for list_id, t_list in enumerate(dataset.target_list):
            target = torch.unique(t_list)[0].item()
            self.class_to_list_ids[target].append(list_id)

    def __iter__(self):
        # Step 0: Prepare the original and working lists of indices
        list_id_to_indices_ref = {
            list_id: list(index_range)
            for list_id, index_range in self.dataset._index_range_dict.items()
        }
        class_to_list_ids = copy.deepcopy(self.class_to_list_ids)

        # Debug: Print at the beginning of each epoch
        # print("\n--- New Epoch ---")

        # Step 1: Group samples by list_id based on `_index_range_dict`
        # Each list_id is mapped to a list of sample indices in its range
        list_id_to_indices = copy.deepcopy(list_id_to_indices_ref)

        # Step 2: Shuffle indices within each list_id group to ensure randomness
        for indices in list_id_to_indices.values():
            random.shuffle(indices)

        # Step 3: Generate the sequence of indices while respecting the max number of unique list_ids per batch
        batches = []
        # available_list_ids = list(list_id_to_indices.keys())
        available_classes = list(class_to_list_ids.keys())

        # Debug: Confirm reset state
        # print("Available classes at epoch start:", available_classes)
        # print("Initial length of list_id_to_indices:")
        # for list_id, indices in list_id_to_indices.items():
        #     print(f"List ID {list_id}: {len(indices)} indices")

        while available_classes:
            # print(f"Available classes: {available_classes}")
            # Randomly select up to `max_list_ids_per_batch` unique list_ids
            selected_classes = random.sample(
                available_classes,
                min(self.max_list_ids_per_batch, len(available_classes)),
            )

            selected_list_ids = []
            for activity_class in selected_classes:
                available_list_ids = [
                    list_id
                    for list_id in class_to_list_ids[activity_class]
                    if list_id_to_indices[list_id]
                ]
                if available_list_ids:
                    selected_list_ids.append(random.choice(available_list_ids))
            # Debug: Check selected classes and list IDs
            # print("Selected classes:", selected_classes)
            # print("Selected list_ids:", selected_list_ids)

            # If selected_list_ids is unexpectedly empty, print a warning and break
            # if not selected_list_ids:
            #     print(
            #         "Warning: selected_list_ids is empty. Check list_id_to_indices and available_classes."
            #     )
            #     print("Available classes:", available_classes)
            #     print("List_id_to_indices content:")
            #     for list_id, indices in list_id_to_indices.items():
            #         print(f"List ID {list_id}: {len(indices)} remaining indices")
            #     break

            if len(selected_list_ids) == 0:
                break

            # print(f"Selected list_ids: {selected_list_ids}")
            # Collect indices for the selected list_ids to form a batch
            total_remaining_samples = sum(
                len(list_id_to_indices[list_id]) for list_id in selected_list_ids
            )
            batch_indices = []
            for list_id in selected_list_ids:
                remaining_samples = len(list_id_to_indices[list_id])
                take_count = max(
                    1,
                    int(
                        self.batch_size * (remaining_samples / total_remaining_samples)
                    ),
                )
                # Ensure we don’t exceed available samples
                take_count = min(take_count, len(list_id_to_indices[list_id]))
                batch_indices.extend(list_id_to_indices[list_id][:take_count])
                list_id_to_indices[list_id] = list_id_to_indices[list_id][take_count:]

                # Remove list_id from available_list_ids if no more samples remain
                if not list_id_to_indices[list_id]:
                    # Loop through each activity class and find the class that contains the depleted list_id
                    for activity_class, list_ids in class_to_list_ids.items():
                        if list_id in list_ids:
                            # Remove the depleted list_id from its class's list_ids
                            list_ids.remove(list_id)
                            depleted_class = activity_class
                            break
                    # print(f"List_id {list_id} is depleted of class {depleted_class}")
                    # print(
                    #     f"remaining no of {depleted_class}: ",
                    #     len(self.class_to_list_ids[depleted_class]),
                    # )
                    # If the activity class no longer has any list_ids with samples, remove it from available_classes
                    if depleted_class and not class_to_list_ids[depleted_class]:
                        available_classes.remove(depleted_class)

            # Adjust batch size if necessary by trimming or duplicating selected samples
            if len(batch_indices) < self.batch_size:
                # If we have not reached the batch size, we need to add more list_ids
                if len(selected_list_ids) < self.max_list_ids_per_batch:
                    # Add more list_ids to the batch
                    additional_list_ids = random.sample(
                        list(list_id_to_indices_ref.keys()),
                        self.max_list_ids_per_batch - len(selected_list_ids),
                    )
                    selected_list_ids.extend(additional_list_ids)

                additional_indices = list(
                    chain.from_iterable(
                        list_id_to_indices_ref[list_id] for list_id in selected_list_ids
                    )
                )
                random.shuffle(additional_indices)
                batch_indices.extend(
                    additional_indices[: self.batch_size - len(batch_indices)]
                )

            # assert (
            #     len(batch_indices) == self.batch_size
            # ), f"Batch size is {len(batch_indices)}"

            # Shuffle and add to batches
            random.shuffle(batch_indices)
            batches.append(batch_indices)

        # Step 4: Shuffle the batches to add randomness across the epoch and return the iterator
        # random.shuffle(batches)
        return iter(chain.from_iterable(batches))

    def __len__(self):
        return sum(len(indices) for indices in self.dataset._index_range_dict.values())


class WIMUSimCustomDataLoader(DataLoader):
    # This is for additional experiments with WIMUSim paper.
    dataset: WIMUSimDataset

    def __init__(
        self,
        dataset: WIMUSimDataset,
        aug_planner,
        H,
        sample: bool = True,
        n_batches_per_epoch=None,
        *args,
        **kwargs,
    ):
        super(WIMUSimCustomDataLoader, self).__init__(dataset, *args, **kwargs)
        self.aug_planner: AugPolicyPlanner = aug_planner
        self.sample = sample  # Whether to sample a new sub-policy for each batch
        self.wimusim_env = None
        self.device = dataset.device
        self.n_batches_per_epoch = n_batches_per_epoch
        self.H = H  # Use this H always.
        self._init_wimusim_env()

    def __len__(self):
        if self.n_batches_per_epoch is not None:
            return self.n_batches_per_epoch  # Use specified number of batches
        else:
            return super().__len__()

    def _init_wimusim_env(self):
        self.wimusim_env = wimusim.WIMUSim(
            B=self.dataset.B_list[0],
            D=self.dataset.D_list[0],
            P=self.dataset.P_list[0],
            H=self.dataset.H_list[0],
            device=self.dataset.device,
        )

    def gen_P(self, B):
        return wimusim.WIMUSim.Placement(
            rp={
                ("BELLY", "BACK"): np.array([0.0, -0.10, 0.20]),
                ("L_SHOULDER", "LUA"): np.array([0.15, 0.00, 0.05]),
                ("L_ELBOW", "LLA"): np.array([0.15, 0.00, 0.05]),
                ("L_KNEE", "L-SHOE"): np.array(
                    [0.0, 0.0, 0.0] + B.rp[("L_KNEE", "L_ANKLE")].detach().cpu().numpy()
                ),
            },
            ro={
                # ('BELLY', 'BACK'): np.deg2rad(np.array([-90, 0., 90])),
                ("BELLY", "BACK"): np.deg2rad(
                    np.array([90, 0.0, -90])
                ),  # or [-90, 0, 90]
                ("L_SHOULDER", "LUA"): np.deg2rad(np.array([90.0, 0.0, 180.0])),
                ("L_ELBOW", "LLA"): np.deg2rad(np.array([90.0, 0.0, 180.0])),
                ("L_KNEE", "L-SHOE"): np.deg2rad(
                    np.array([180, 0.0, -90])
                ),  # or [0., 180, 0] # Z-down (most likely)
            },
            device="cuda:0",
        )

    def __iter__(self):
        # Create an iterator for the base DataLoader
        base_iter = super(WIMUSimCustomDataLoader, self).__iter__()
        batch_count = 0

        while True:
            try:
                # Get the next batch from the base iterator
                X_d, y, (idx, list_ids) = next(base_iter)
            except StopIteration:
                if self.n_batches_per_epoch is None:
                    break
                # Restart the base iterator when data is exhausted
                base_iter = super(WIMUSimCustomDataLoader, self).__iter__()
                X_d, y, (idx, list_ids) = next(base_iter)

            # list_ids specifies which P, B, H to use for the current batch

            # Sample sub-policy based on a_subpolicies unless one_hot is given
            # For training dataset, we want to sample a new sub-policy for each batch
            # For validation dataset, we want to use the same sub-policy as the training batch
            if self.sample:
                self.aug_planner.sample_subpolicy()

            virtual_IMU_dict = {
                imu_name: {"acc": [], "gyro": []}
                for imu_name in self.dataset.P_list[0].imu_names
            }
            y_list = []
            idx_list = []
            list_id_list = []
            # We can use the same TimeScaling
            for augmentor in self.aug_planner.current_subpolicy:
                # TimeScaling should be applied to the data.
                if isinstance(augmentor, ppda.TimeScaling):
                    # Sample a scaling factor and re-fetch data with the new scale
                    scale_factor: float = augmentor.sample_scaling_factor()
                    T_orig = X_d.shape[2]  # T is the time dimension (N, J, T, C)
                    if scale_factor > 1.0:
                        # Add this line to prevent index out of bounds (possibly due to some error in the CustomSampler)
                        idx = torch.clamp(idx, 0, X_d.shape[0] - 1)
                        # Re-fetch data with the scaled window size
                        # Loading the entire dataset is faster than loading one by one with randomized indices
                        X_d = torch.stack(
                            [
                                self.dataset.__getitem__(i, scale=scale_factor)[0]
                                for i in range(self.dataset.__len__())
                            ]
                        )[idx]
                    # Apply time scaling if needed
                    X_d = augmentor.apply_augmentation(X=X_d, T_orig=T_orig)
                elif isinstance(augmentor, ppda.TimeWarping):
                    X_d = augmentor.apply_augmentation(X=X_d)
                elif isinstance(augmentor, ppda.MagnitudeWarping):
                    X_d = augmentor.apply_augmentation(X=X_d)

            # Data Augmentation involving B, P, H are applied here
            for list_id in torch.unique(list_ids):
                # print(f"processing list_id {list_id}...")
                X_d_sbj = X_d[list_ids == list_id]  # Get the data of the subject.
                y_sbj = y[list_ids == list_id]  # Get the target of the subject.
                idx_sbj = idx[list_ids == list_id]  # Get the index of the subject.

                B = copy.deepcopy(self.dataset.B_list[list_id])
                P = copy.deepcopy(self.gen_P(B))
                H = copy.deepcopy(self.H)
                # TODO: We don't really need to use this as it will be overwritten by the WIMUSim object

                # We can use the same TimeScaling
                for augmentor in self.aug_planner.current_subpolicy:
                    if isinstance(augmentor, ppda.Rotation):
                        if augmentor.paramix:
                            # Select P randomly from the P_list if paramix is enabled
                            P = copy.deepcopy(random.choice(self.dataset.P_list))
                        P = augmentor.apply_augmentation(P=P)
                    elif isinstance(augmentor, ppda.NoiseBias):
                        if hasattr(H, "imu_names"):
                            H = augmentor.apply_augmentation(H=H)
                        else:
                            # Use P.imu_names if H does not have imu_names
                            H = augmentor.apply_augmentation(H=H, imu_names=P.imu_names)

                # Transform D_data to WIMUSim.Dynamics
                # It's okay to apply modification directly to the D as its newly created.
                D = wimusim.WIMUSim.Dynamics(
                    translation={"XYZ": X_d_sbj[:, 0, :, 1:].detach().cpu().numpy()},
                    orientation={
                        joint_name: X_d_sbj[:, joint_id, :, :].detach().cpu().numpy()
                        for joint_name, joint_id in self.dataset._D_ori_key_idx.items()
                    },
                    sample_rate=25,
                    device=self.device,
                )

                # Apply the augmentations from the current subpolicy
                # for augmentor in self.aug_planner.current_subpolicy:
                #    # Apply data augmentation here
                for augmentor in self.aug_planner.current_subpolicy:
                    if isinstance(augmentor, ppda.MagnitudeScaling):
                        D = augmentor.apply_augmentation(D=D)

                if self.wimusim_env is None:
                    self.wimusim_env = wimusim.WIMUSim(B, D, P, H, device=self.device)
                else:
                    # Reuse the WIMUSim object if it already exists
                    self.wimusim_env.B = B
                    self.wimusim_env.D = D
                    self.wimusim_env.P = P
                    self.wimusim_env.H = H

                virtual_IMU_dict_sbj = self.wimusim_env.simulate(mode="generate")
                print(virtual_IMU_dict_sbj.keys())
                for imu_name, imu_data in virtual_IMU_dict_sbj.items():
                    virtual_IMU_dict[imu_name]["acc"].append(imu_data[0])
                    virtual_IMU_dict[imu_name]["gyro"].append(imu_data[1])
                y_list.append(y_sbj)
                idx_list.append(idx_sbj)
                list_id_list.append(list_ids[list_ids == list_id])

            if self.dataset.acc_only:
                X_res = torch.concat(
                    [
                        torch.concat(virtual_IMU_dict[imu_name]["acc"], dim=0)
                        for imu_name in self.dataset.P_list[0].imu_names
                    ],
                    dim=-1,
                )
            elif self.dataset.gyro_only:
                X_res = torch.concat(
                    [
                        torch.concat(virtual_IMU_dict[imu_name]["gyro"], dim=0)
                        for imu_name in self.dataset.P_list[0].imu_names
                    ],
                    dim=-1,
                )
            else:
                if self.dataset.data_order == "alternate":
                    X_res = torch.concat(
                        [
                            torch.concat(virtual_IMU_dict[imu_name][sensor_type], dim=0)
                            for imu_name in self.dataset.P_list[0].imu_names
                            for sensor_type in ["acc", "gyro"]
                        ],
                        dim=-1,
                    )
                elif self.dataset.data_order == "sequential":
                    X_res = torch.concat(
                        [
                            torch.concat(virtual_IMU_dict[imu_name][sensor_type], dim=0)
                            for sensor_type in ["acc", "gyro"]
                            for imu_name in self.dataset.P_list[0].imu_names
                        ],
                        dim=-1,
                    )
                else:
                    raise ValueError("Invalid order", self.dataset.data_order)

            y = torch.concat(y_list, dim=0).to(self.device)
            idx = torch.concat(idx_list, dim=0)
            list_ids = torch.concat(list_id_list, dim=0)

            # Apply scaling (standardization)
            X = (
                (
                    (X_res.detach().cpu() - self.dataset.scale_config["mean"])
                    / self.dataset.scale_config["std"]
                )
                .to(self.device)
                .float()
            )

            # Multiply X by the one-hot vector for proper gradient flow
            # This ensures one-hot_vector is still involved in the computation
            # This won't change the value of X
            for e in self.aug_planner.current_subpolicy_one_hot:
                if e == 1.0:
                    X = e * X

            yield X, y, (idx, list_ids)
            batch_count += 1

            if (
                self.n_batches_per_epoch is not None
                and batch_count >= self.n_batches_per_epoch
            ):
                break
