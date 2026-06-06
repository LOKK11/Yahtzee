import torch
from itertools import combinations_with_replacement


class YahtzeeCategory:
    def __init__(self, batch_size, category, device="cpu"):
        self.n = batch_size
        self.device = device
        self.category = category
        self.REWARD_EXPONENT = 2

        # Game Constants
        self.NUM_DICE = 5
        self.DICE_FACES = 6
        self.max_scores = torch.tensor(
            [
                5,  # Ones
                10,  # Twos
                15,  # Threes
                20,  # Fours
                25,  # Fives
                30,  # Sixes
                12,  # Two of a kind
                22,  # Two pairs
                18,  # Three of a kind
                24,  # Four of a kind
                28,  # Full house
                15,  # Small straight
                20,  # Large straight
                50,  # Yahtzee
                30,  # Chance
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self.keep_combinations = []
        for r in range(5):
            for combo in combinations_with_replacement(range(1, 7), r):
                counts = torch.zeros(7, dtype=torch.long, device=self.device)
                for val in combo:
                    counts[val] += 1
                self.keep_combinations.append(counts)

        # Full keep (5 dice)
        self.keep_combinations.append(
            torch.fill(torch.zeros(7, dtype=torch.long, device=self.device), 5)
        )

        self.keep_combinations = torch.stack(self.keep_combinations)

        self.ROLL_ACTIONS = len(self.keep_combinations)

        self.reset()

    def reset(self):
        """Resets all games in the batch."""
        self.dice = torch.randint(
            1, 7, size=(self.n, self.NUM_DICE), device=self.device
        )

        self.rolls_left = 2
        # Mask for completed games
        self.finished = False

    def get_potential_scores(self):
        """
        Calculates the score for every category for the current dice state.
        Returns: (N, 15) array of scores.
        """
        dice_one_hot = torch.nn.functional.one_hot(
            self.dice, num_classes=7
        )  # (N, 5, 7)
        counts = dice_one_hot.sum(dim=1)  # (N, 7)

        scores = torch.zeros((self.n, 15), dtype=torch.float32, device=self.device)

        # --- Upper Section ---

        # Ones to Sixes (Indices 0-5)
        for i in range(1, 7):
            scores[:, i - 1] = counts[:, i] * i

        # --- Lower Section ---

        # Helper: Sum of all dice
        dice_sums = self.dice.sum(dim=1)

        # Two of a kind (Index 6)
        tok_score = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        for val in range(1, 7):
            mask = counts[:, val] >= 2
            tok_score[mask] = val * 2
        scores[:, 6] = tok_score

        # Two pairs (Index 7)
        pair_mask = counts[:, 1:] >= 2
        num_pairs = pair_mask.sum(dim=1)

        # To get the sum of values that are pairs:
        values_arr = torch.tile(torch.arange(1, 7, device=self.device), (self.n, 1))
        pair_vals_sum = (values_arr * pair_mask).sum(dim=1)
        scores[:, 7] = torch.where(num_pairs >= 2, pair_vals_sum * 2, 0)

        # Three of a kind (Index 8)
        thok_score = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        for val in range(1, 7):
            mask = counts[:, val] >= 3
            thok_score[mask] = val * 3
        scores[:, 8] = thok_score

        # Four of a kind (Index 9)
        fok_score = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        for val in range(1, 7):
            mask = counts[:, val] >= 4
            fok_score[mask] = val * 4
        scores[:, 9] = fok_score

        # Full House (Index 10)
        has_3 = (counts[:, 1:] == 3).any(dim=1)
        has_2 = (counts[:, 1:] == 2).any(dim=1)
        scores[:, 10] = torch.where(has_3 & has_2, dice_sums, 0)

        # Small Straight (Index 11)
        ss_mask = (counts[:, 1:6] == 1).all(dim=1)
        scores[:, 11] = torch.where(ss_mask, 15, 0)
        # Large Straight (Index 12)
        ls_mask = (counts[:, 2:7] == 1).all(dim=1)
        scores[:, 12] = torch.where(ls_mask, 20, 0)

        # Yahtzee (Index 13)
        c = counts[:, 1:]
        y_mask = (c == 5).any(dim=1)
        base_scores = torch.where(y_mask, 50.0, 0.0)
        max_counts, _ = c.max(dim=1, keepdim=True)
        is_max_count = c == max_counts
        faces = torch.arange(1, 7, device=counts.device, dtype=c.dtype)
        best_faces = (is_max_count * faces).max(dim=1).values
        bonus = best_faces / 5
        scores[:, 13] = base_scores + bonus

        # Chance (Index 14)
        scores[:, 14] = dice_sums

        return scores

    def step(self, actions):
        """
        Advances the game state based on actions.
        actions: (N,) int array of action indices.
        Returns:
            states (tensor),
            rewards (float array),
            dones (bool array),
            infos (dict)
        """
        rewards = torch.zeros(self.n, dtype=torch.float32, device=self.device)

        # --- Handle Roll Actions ---
        if self.rolls_left > 0:
            target_counts = self.keep_combinations[actions]

            current_dice = self.dice
            hold_masks = torch.zeros_like(current_dice, dtype=torch.bool)

            for v in range(1, 7):
                v_needed = target_counts[:, v]

                is_v = current_dice == v

                # Use cumulative sum to identify the 1st, 2nd, 3rd... instance of value 'v'
                v_rank = torch.cumsum(is_v.to(torch.long), dim=1)

                # Keep the die if it is value 'v' AND its rank is <= number we need
                hold_masks |= is_v & (v_rank <= v_needed.unsqueeze(1))

            new_rolls = torch.randint(1, 7, size=current_dice.shape, device=self.device)
            self.dice = torch.where(hold_masks, current_dice, new_rolls)
            self.rolls_left -= 1

        # --- Handle Category Actions ---
        else:
            # Calculate scores for current dice
            rewards = self.get_potential_scores()[:, self.category]

        average_reward = rewards.mean().item()
        rewards = rewards**self.REWARD_EXPONENT

        return rewards, average_reward

    def get_action_masks(self, category):
        total_actions = self.ROLL_ACTIONS
        mask = torch.zeros(
            (self.n, total_actions), dtype=torch.bool, device=self.device
        )

        # Roll actions: Calculate current dice counts in hand
        dice_one_hot = torch.nn.functional.one_hot(self.dice, num_classes=7)
        hand_counts = dice_one_hot.sum(dim=1)  # (N, 7)

        mask = (self.keep_combinations.unsqueeze(0) <= hand_counts.unsqueeze(1)).all(
            dim=2
        )
        allow_keep_all = torch.tensor(
            [
                1,  # Ones
                1,  # Twos
                1,  # Threes
                1,  # Fours
                1,  # Fives
                1,  # Sixes
                0,  # Two of a kind
                0,  # Two pairs
                0,  # Three of a kind
                0,  # Four of a kind
                1,  # Full house
                1,  # Small straight
                1,  # Large straight
                1,  # Yahtzee
                1,  # Chance
            ]
        )
        mask[:, -1] = allow_keep_all[category]

        return mask

    def get_encoded_state(self):
        """
        Returns the state tensor compatible with YahtzeeNetCategory.
        Shape: (N, 44)
        """
        encoded = torch.zeros((self.n, 44), dtype=torch.float32, device=self.device)

        # 1. Dice Values (41 inputs)
        dice_one_hot = torch.nn.functional.one_hot(self.dice - 1, num_classes=6)
        counts = dice_one_hot.sum(dim=1)
        highest_counts, _ = counts.max(dim=1)

        for val_idx in range(6):
            val_counts = counts[:, val_idx].long()
            row_indices = torch.arange(self.n, device=self.device)
            col_indices = val_idx * 6 + val_counts
            encoded[row_indices, col_indices] = 1.0

        encoded[torch.arange(self.n), 35 + highest_counts] = 1.0

        # 2. Rolls Left (3 inputs)
        rl = self.rolls_left
        encoded[torch.arange(self.n), 41 + rl] = 1.0

        return encoded
