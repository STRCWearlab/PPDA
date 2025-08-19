import torch
from torch.nn import functional as F
from simuaug.utils import get_default_device, paint


class AugPolicyPlanner:
    def __init__(
        self,
        sub_policies,
        aug_params,
        a_subpolicies=None,
        tau=1.0,
        tau_decay=1e-2,
        min_tau=1.0,
        device=None,
    ):
        if device is None:
            self.device = get_default_device()
        else:
            self.device = device

        self.sub_policies = sub_policies
        if a_subpolicies is None:
            self.a_subpolicies = torch.ones(len(sub_policies), device=device)
        else:
            self.a_subpolicies = a_subpolicies.to(device)

        assert len(self.sub_policies) == len(
            self.a_subpolicies
        ), "Number of sub-policies and probabilities must match."

        self.aug_params = aug_params
        self.tau_0 = tau
        self.tau = tau
        self.tau_decay = tau_decay
        self.min_tau = min_tau
        self.current_subpolicy = None  # Current policy to be set
        self.current_subpolicy_one_hot = None  # Current policy in one-hot encoding
        # Initialize current_subpolicy and current_subpolicy_one_hot
        self.sample_subpolicy()

        print(
            paint(
                f"Initialized AugPolicyPlanner with {len(sub_policies)} sub-policies and default sampling probabilities: {a_subpolicies.detach().cpu().tolist()}",
                "green",
            )
        )
        for i, sub_policy in enumerate(sub_policies):
            formatted_policy = ", ".join([str(aug) for aug in sub_policy])
            print(
                paint(
                    f"Policy {i+1}: {formatted_policy}",
                    "green",
                )
            )

    def update_tau(self, epoch):
        """
        Update the temperature tau based on the epoch using the formula:
        tau = tau * exp(-0.05 * epoch)

        :param epoch: Current training epoch
        """
        # AutoAugHAR use the following configuration
        self.tau = max(
            self.tau_0 * torch.exp(torch.tensor(-0.05 * epoch)).item(), self.min_tau
        )
        # self.tau = self.tau_0 * torch.exp(torch.tensor(-0.05 * epoch)).item()

    def reinit_a_subpolicies(self, value=1e-3):
        """
        Initialize the a_subpolicies with the given value.
        """
        self.a_subpolicies.detach().fill_(value).requires_grad_(True)

    def init_tau(self, value=None):
        """
        Initialize the temperature tau with the given value.
        """
        if value is None:
            self.tau = self.tau_0
        else:
            self.tau = value

    def sample_subpolicy(
        self,
    ):
        """
        Sample a subpolicy from the list of subpolicies.

        :return: A subpolicy to be applied.
        """
        self.current_subpolicy_one_hot = F.gumbel_softmax(
            self.a_subpolicies, tau=self.tau, hard=True, dim=0
        )
        self.current_subpolicy = self.sub_policies[
            self.current_subpolicy_one_hot.argmax()
        ]
        return

    @property
    def current_subpolicy(self):
        return self._current_subpolicy

    @current_subpolicy.setter
    def current_subpolicy(self, policy):
        self._current_subpolicy = policy

    @property
    def current_subpolicy_one_hot(self):
        return self._current_subpolicy_one_hot

    @current_subpolicy_one_hot.setter
    def current_subpolicy_one_hot(self, policy_one_hot):
        self._current_subpolicy_one_hot = policy_one_hot

    @property
    def aug_params(self):
        return self._aug_params

    @aug_params.setter
    def aug_params(self, aug_params):
        self._aug_params = aug_params
