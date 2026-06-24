from torch import nn
import config as cfg


class RollPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.common_stream = nn.Sequential(
            nn.Linear(
                cfg.ROLL_SPECIFIC_INPUTS + cfg.COMMON_INPUTS,
                512,
            ),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, cfg.ROLL_ACTIONS),
        )

    def forward(self, x):
        common_features = self.common_stream(x)
        value = self.value_stream(common_features)
        advantages = self.advantage_stream(common_features)

        # Q = V + (A - mean(A))
        q_vals = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_vals


class CategoryPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.common_stream = nn.Sequential(
            nn.Linear(
                cfg.COMMON_INPUTS,
                512,
            ),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, cfg.CATEGORY_ACTIONS),
        )

    def forward(self, x):
        common_features = self.common_stream(x)
        value = self.value_stream(common_features)
        advantages = self.advantage_stream(common_features)

        # Q = V + (A - mean(A))
        q_vals = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_vals
