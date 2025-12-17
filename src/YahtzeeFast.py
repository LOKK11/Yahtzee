import torch


class YahtzeeFast:
    def __init__(self, batch_size, device="cpu"):
        self.n = batch_size
        self.device = device

        # Game Constants
        self.NUM_DICE = 5
        self.DICE_FACES = 6
        self.NUM_CATEGORIES = 16  # Includes Bonus
        self.CATEGORY_NAMES = [
            "ones",
            "twos",
            "threes",
            "fours",
            "fives",
            "sixes",
            "two_of_a_kind",
            "two_pairs",
            "three_of_a_kind",
            "four_of_a_kind",
            "full_house",
            "small_straight",
            "large_straight",
            "yahtzee",
            "chance",
            "bonus",
        ]

        # Action Constants
        self.CAT_ACTIONS = 15
        self.ROLL_ACTIONS = 32

        # Precompute roll action masks
        self.roll_action_masks = torch.zeros(
            (32, 5), dtype=torch.bool, device=self.device
        )
        for i in range(self.ROLL_ACTIONS):
            for bit in range(5):
                if (i >> bit) & 1:
                    self.roll_action_masks[i, bit] = True

        self.reset()

    def reset(self):
        """Resets all games in the batch."""
        self.dice = torch.randint(
            1, 7, size=(self.n, self.NUM_DICE), device=self.device
        )

        self.scores = torch.full(
            (self.n, self.NUM_CATEGORIES), -1, dtype=torch.float32, device=self.device
        )

        self.rolls_left = 2
        # Mask for completed games
        self.finished = False

    def roll_dice(self, hold_masks):
        """
        Rolls dice for all games.
        hold_masks: (N, 5) boolean array. True = Keep die, False = Reroll.
        """
        new_rolls = torch.randint(
            1, 7, size=(self.n, self.NUM_DICE), device=self.device
        )
        self.dice = torch.where(hold_masks, self.dice, new_rolls)
        self.rolls_left -= 1

    def get_potential_scores(self):
        """
        Calculates the score for every category for the current dice state.
        Returns: (N, 16) array of scores.
        """
        dice_one_hot = torch.nn.functional.one_hot(
            self.dice, num_classes=7
        )  # (N, 5, 7)
        counts = dice_one_hot.sum(dim=1)  # (N, 7)

        scores = torch.zeros(
            (self.n, self.NUM_CATEGORIES), dtype=torch.float32, device=self.device
        )

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
        y_mask = (counts[:, 1:] == 5).any(dim=1)
        scores[:, 13] = torch.where(y_mask, 50, 0)
        # Chance (Index 14)
        scores[:, 14] = dice_sums

        # Bonus (Index 15)
        # Always 0 here, handled in state update

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
        is_category_action = actions[0] < self.CAT_ACTIONS
        is_roll_action = actions[0] >= self.CAT_ACTIONS

        rewards = torch.zeros(self.n, dtype=torch.float32, device=self.device)

        # --- Handle Roll Actions ---
        if is_roll_action:
            # Convert action ID to 0-31 range
            roll_action_ids = actions - self.CAT_ACTIONS

            masks = self.roll_action_masks[roll_action_ids]

            new_rolls = torch.randint(
                1, 7, size=(len(actions), self.NUM_DICE), device=self.device
            )

            # Apply hold mask
            updated_dice = torch.where(masks, self.dice, new_rolls)
            self.dice = updated_dice

            self.rolls_left -= 1

        # --- Handle Category Actions ---
        elif is_category_action:
            bonuses_before = torch.where(
                self.scores[:, 15] == -1, 0, self.scores[:, 15]
            )

            # Calculate scores for current dice
            all_scores = self.get_potential_scores()  # (N, 16)

            earned_scores = all_scores[torch.arange(self.n), actions]
            self.scores[torch.arange(self.n), actions] = earned_scores
            rewards = earned_scores

            # --- Bonus Logic ---
            # Check upper section sum (indices 0-5)
            upper_scores = self.scores[:, 0:6]
            upper_sums = torch.where(upper_scores > -1, upper_scores, 0).sum(dim=1)

            bonus_awarded = upper_sums >= 63
            self.scores[:, 15] = torch.where(bonus_awarded, 50, -1)
            bonuses_after = torch.where(self.scores[:, 15] == -1, 0, self.scores[:, 15])
            rewards += bonuses_after - bonuses_before

            # Reset Dice and Rolls for next turn
            self.rolls_left = 3
            self.dice = torch.randint(
                1, 7, size=(self.n, self.NUM_DICE), device=self.device
            )
            self.rolls_left -= 1

        # --- Check Game Over ---
        # Game over if all first 15 categories (0-14) are played (score != -1)
        played_count = (self.scores[0, 0:15] != -1).sum()
        finished = played_count == 15

        if finished:
            dones = torch.ones(self.n, dtype=torch.float32, device=self.device)
            average_final_score = self.scores.sum(dim=1, dtype=torch.float32).mean()
        else:
            dones = torch.zeros(self.n, dtype=torch.float32, device=self.device)
            average_final_score = 0.0

        return rewards, dones, average_final_score

    def get_action_masks(self):
        """
        Returns boolean mask (N, ACTION_SIZE)
        True = Valid action, False = Invalid
        """
        total_actions = self.CAT_ACTIONS + self.ROLL_ACTIONS
        mask = torch.zeros(
            (self.n, total_actions), dtype=torch.bool, device=self.device
        )

        can_roll = self.rolls_left > 0
        if not can_roll:
            # 1. Category Actions (0-14)
            mask[:, :15] = self.scores[:, :15] == -1
        else:
            # 2. Roll Actions (15-46)
            mask[:, 15:47] = True

        return mask

    def get_encoded_state(self):
        """
        Returns the state tensor compatible with YahtzeeNet.
        Shape: (N, 62)
        """
        encoded = torch.zeros((self.n, 62), dtype=torch.float32, device=self.device)

        # 1. Dice Values (30 inputs)
        pos_indices = torch.tile(torch.arange(5, device=self.device), (self.n, 1))
        val_offsets = (self.dice - 1) * 5
        flat_indices = val_offsets + pos_indices
        row_indices = torch.arange(self.n, device=self.device)[:, torch.newaxis]
        encoded[row_indices, flat_indices] = 1.0

        # 2. Categories Played (16 inputs)
        is_played = (self.scores != -1).to(torch.float32)
        encoded[:, 30:46] = is_played

        # 3. Upper Section Score (13 inputs)
        upper_vals = self.scores[:, 0:6]
        upper_score = torch.where(upper_vals > -1, upper_vals, 0).sum(dim=1)
        upper_score = torch.clamp(upper_score, 0, 63) // (63 / 12)
        mask = torch.zeros((self.n, 13), dtype=torch.float32, device=self.device)
        mask[torch.arange(self.n), upper_score.to(torch.int32)] = 1.0
        encoded[:, 46:59] = mask

        # 4. Rolls Left (3 inputs)
        rl = self.rolls_left
        encoded[torch.arange(self.n), 59 + rl] = 1.0

        return encoded
