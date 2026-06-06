import torch
from itertools import combinations_with_replacement


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
        self.CAT_ACTIONS = 15

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

    def get_final_scores(self):
        return torch.where(self.scores == -1, 0, self.scores).sum(
            dim=1, dtype=torch.float32
        )

    def get_average_final_score(self):
        return (
            torch.where(self.scores == -1, 0, self.scores)
            .sum(dim=1, dtype=torch.float32)
            .mean()
            .item()
        )

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

    def step(self, actions, action_type):
        """
        Advances the game state based on actions.
        actions: (N,) int array of action indices.
        Returns:
            rewards: (N,) float array of rewards received from the action.
        """
        rewards = None

        # --- Handle Roll Actions ---
        if action_type == "roll":
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
        elif action_type == "category":
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
            self.rolls_left = 2
            self.dice = torch.randint(
                1, 7, size=(self.n, self.NUM_DICE), device=self.device
            )

        return rewards

    def get_category_action_masks(self):
        mask = self.scores[:, :15] == -1
        if self.rolls_left > 0:
            mask[:, :14] = True
            mask[:, 14] = False

        return mask

    def get_roll_action_masks(self):
        dice_one_hot = torch.nn.functional.one_hot(self.dice, num_classes=7)
        hand_counts = dice_one_hot.sum(dim=1)  # (N, 7)

        mask = (self.keep_combinations.unsqueeze(0) <= hand_counts.unsqueeze(1)).all(
            dim=2
        )
        mask[:, -1] = True  # Always allow full keep

        return mask

    def get_encoded_state(self):
        """
        Returns the state tensor compatible with YahtzeeNet.
        Shape: (N, 84)
        """
        encoded = torch.zeros((self.n, 84), dtype=torch.float32, device=self.device)

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

        # 3. Normalized score yields (15 inputs)
        not_played = (self.scores[:, :15] == -1).to(torch.float32)
        all_scores = self.get_potential_scores()[:, :15] * not_played
        normalized_scores = all_scores / self.max_scores
        encoded[:, 44:59] = normalized_scores

        # 4. Would yield bonus (6 inputs)
        would_yield_bonus = torch.zeros(
            (self.n, 6), dtype=torch.float32, device=self.device
        )
        for i in range(6):
            upper_vals = self.scores[:, 0:6].clone()
            mask = torch.where(upper_vals > -1, upper_vals, 0).sum(dim=1) < 63
            mask &= upper_vals[:, i] == -1
            upper_vals[:, i] = torch.where(
                mask,
                self.get_potential_scores()[:, i],
                upper_vals[:, i],
            )
            new_upper_score = torch.where(upper_vals > -1, upper_vals, 0).sum(dim=1)
            would_yield_bonus[:, i] = ((new_upper_score >= 63) & mask).to(torch.float32)
        encoded[:, 59:65] = would_yield_bonus

        # 5. Categories Played (16 inputs)
        is_played = (self.scores != -1).to(torch.float32)
        encoded[:, 65:81] = is_played

        # 6. Normalized Upper Section Score (1 input)
        upper_vals = self.scores[:, 0:6]
        upper_score = torch.where(upper_vals > -1, upper_vals, 0).sum(dim=1)
        encoded[:, 81] = torch.clamp(upper_score, max=63) / 63.0

        # 7. Normalized Categories Played (1 input)
        categories_played = (self.scores[:, :15] != -1).sum(dim=1)
        encoded[:, 82] = categories_played / 15.0

        # 8. Bonus available (1 input)
        max_possible_upper_score = torch.where(
            upper_vals > -1, upper_vals, self.max_scores[0:6]
        ).sum(dim=1)
        bonus_available = (upper_score < 63) & (max_possible_upper_score >= 63)
        encoded[:, 83] = bonus_available.to(torch.float32)
        return encoded

    @torch.no_grad()
    def analyze_batch_stats(
        self, all_roll_actions, all_category_actions, all_rewards, max_score_bins=51
    ):
        """
        Vectorized calculation of game statistics.
        all_actions: Tensor of all actions taken in self-play (Steps * N)
        all_rewards: Tensor of all rewards received (Steps * N)
        """
        stats = {}

        # 1. Roll Action Distribution (Actions 15-end)
        stats["roll_counts"] = torch.bincount(
            all_roll_actions, minlength=self.ROLL_ACTIONS
        ).cpu()

        # 2. Category Reward Distributions (Actions 0-14)
        cat_stats = []
        for i in range(15):
            cat_mask = all_category_actions == i
            cat_rewards = all_rewards[cat_mask].long()
            if i < 6:
                cat_rewards = torch.where(
                    cat_rewards > 30, cat_rewards - 50, cat_rewards
                )
            # Clamp to prevent index errors, though rewards should be in range
            cat_rewards = torch.clamp(cat_rewards, 0, max_score_bins - 1)
            dist = torch.bincount(cat_rewards, minlength=max_score_bins).cpu()
            cat_stats.append(dist)
        stats["cat_distributions"] = cat_stats

        # 3. Special Totals
        # Yahtzee is action 13 with reward 50
        stats["yahtzee_count"] = (
            ((all_category_actions == 13) & (all_rewards == 50)).sum().item()
        )
        # Bonus is reward > 50 for upper section actions (0-5)
        stats["bonus_count"] = (
            ((all_category_actions < 6) & (all_rewards > 30)).sum().item()
        )

        return stats
