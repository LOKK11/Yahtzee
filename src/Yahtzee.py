import random
import torch
from YahtzeeAI import (
    RollPolicyNet,
    MultiCategoryNet,
    CategoryPolicyNet,
    load_models,
    ROLL_SPECIFIC_INPUTS,
    DICE_VALUES,
    ROLLS_LEFT,
    CATEGORY_ACTIONS,
)
from YahtzeeFast import YahtzeeFast

MODEL = "models/best_model.pth"
MULTI_MODEL = "models/multi_category_net_2.pth"


class Yahtzee:
    def __init__(self):
        self.device = torch.device("cpu")
        self.fast_env = YahtzeeFast(batch_size=1, device=self.device)

        self.dice = [0] * 5
        self.locked = []
        self.rolls_left = 3
        self.category_names = self.fast_env.CATEGORY_NAMES

        self.categories = {name: None for name in self.category_names}
        self.categories_played = 0
        self.neural_network = False
        self.roll_policy_net = None
        self.category_policy_net = None
        self.multi_category_model = None

    def load_models(self):
        """Loads the latest model architecture."""
        try:
            self.roll_policy_net = RollPolicyNet()
            self.category_policy_net = CategoryPolicyNet()
            self.multi_category_model = MultiCategoryNet()
            load_models(
                self.roll_policy_net,
                self.category_policy_net,
                MODEL,
                device=self.device,
            )
            self.multi_category_model.load_state_dict(
                torch.load(MULTI_MODEL, map_location=self.device)
            )
            self.roll_policy_net.eval()
            self.multi_category_model.eval()
            self.neural_network = True
            print(f"Loaded model: {MODEL}")
            print(f"Loaded multi-category model: {MULTI_MODEL}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def sync_to_fast_env(self):
        """Updates the internal fast_env state to match current game state."""
        # Update Dice
        self.fast_env.dice = torch.tensor(
            [self.dice], dtype=torch.long, device=self.device
        )

        # Update Scores (Map None to -1)
        score_tensor = torch.full((1, 16), -1, dtype=torch.float32, device=self.device)
        for i, name in enumerate(self.category_names):
            val = self.categories[name]
            score_tensor[0, i] = val if val is not None else -1
        self.fast_env.scores = score_tensor

        self.fast_env.rolls_left = self.rolls_left

    def get_ai_prediction(self):
        """Returns a ranked list of all categories and their best roll actions."""
        if (
            not self.roll_policy_net
            or not self.category_policy_net
            or not self.multi_category_model
            or not self.neural_network
        ):
            return ""

        self.sync_to_fast_env()
        if self.rolls_left != 0:
            state = self.fast_env.get_encoded_state()
            policy_net = self.roll_policy_net
        else:
            state = self.fast_env.get_encoded_state()[:, ROLL_SPECIFIC_INPUTS:]
            policy_net = self.category_policy_net
        category_mask = self.fast_env.get_category_action_masks()
        roll_mask = self.fast_env.get_roll_action_masks()

        with torch.no_grad():
            # 1. Get Q-values for all categories
            q_values = policy_net(state)

            # Mask out already played categories
            q_values = q_values.masked_fill(~category_mask, -1e12)

            # 2. Get indices of categories sorted by Q-value (descending)
            sorted_indices = torch.argsort(q_values, descending=True, dim=1)[0]

            results = []
            # 3. For each category, find what the best roll action would be
            for idx in sorted_indices:
                idx_item = idx.item()
                # If Q-value is very low, it's likely masked/invalid
                if q_values[0, idx_item] < -1e11:
                    continue

                cat_name = self.category_names[idx_item].replace("_", " ").title()

                if self.rolls_left == 0:
                    results.append(f"Rank {len(results) + 1}: {cat_name}")
                else:
                    # Query multi-category model for this specific category
                    cat_tensor = torch.tensor([idx_item], device=self.device)
                    roll_q_values = self.multi_category_model(
                        state[:, : DICE_VALUES + ROLLS_LEFT], cat_tensor
                    )
                    roll_q_values = roll_q_values.masked_fill(~roll_mask, -1e12)
                    roll_action = roll_q_values.argmax(dim=1)

                    # Decode roll action
                    target_counts = self.fast_env.keep_combinations[roll_action][0]
                    keep_desc = []
                    for face in range(1, 7):
                        count = target_counts[face].item()
                        if count == 5:
                            keep_desc = ["All"]
                            break
                        if count > 0:
                            keep_desc.append(f"{count}x{face}")

                    keep_str = ", ".join(keep_desc) if keep_desc else "None"
                    results.append(f"{cat_name}: Keep {keep_str}")

        return "\n".join(results)

    def roll_dice(self):
        if self.rolls_left == 3:
            self.dice = [random.randint(1, 6) for _ in range(5)]
            self.rolls_left -= 1
        elif self.rolls_left > 0:
            for i in range(5):
                if i not in self.locked:
                    self.dice[i] = random.randint(1, 6)
            self.rolls_left -= 1
        else:
            print("No rolls left")

    def reset_dice(self):
        """Resets the dice and rolls left"""
        self.dice = [0] * 5  # Visual reset
        self.locked = []
        self.rolls_left = 3

    def lock_dice(self, dice_idx):
        """Locks the dice that the player wants to keep"""
        if dice_idx not in self.locked:
            self.locked.append(dice_idx)

    def unlock_dice(self, dice_idx):
        """Unlocks the dice that the player wants to unlock"""
        if dice_idx in self.locked:
            self.locked.remove(dice_idx)

    def calculate_score(self, category):
        """Place the dices to one of the categories"""
        if category == "ones":
            return self.dice.count(1)
        if category == "twos":
            return self.dice.count(2) * 2
        if category == "threes":
            return self.dice.count(3) * 3
        if category == "fours":
            return self.dice.count(4) * 4
        if category == "fives":
            return self.dice.count(5) * 5
        if category == "sixes":
            return self.dice.count(6) * 6
        if category == "bonus":
            # Recalculate bonus based on actual filled categories
            score = 0
            for cat in ["ones", "twos", "threes", "fours", "fives", "sixes"]:
                val = self.categories.get(cat)
                if val is not None:
                    score += val
            return 50 if score >= 63 else 0

        if category == "two_of_a_kind":
            for i in range(6, 0, -1):
                if self.dice.count(i) >= 2:
                    return 2 * i
            return 0
        elif category == "two_pairs":
            pairs_values = []
            for i in range(6, 0, -1):
                if self.dice.count(i) >= 2:
                    pairs_values.append(i)

            if len(pairs_values) >= 2:
                return sum(pairs_values) * 2
            return 0
        elif category == "three_of_a_kind":
            for i in range(6, 0, -1):
                if self.dice.count(i) >= 3:
                    return 3 * i
            return 0
        elif category == "four_of_a_kind":
            for i in range(6, 0, -1):
                if self.dice.count(i) >= 4:
                    return 4 * i
            return 0
        elif category == "full_house":
            counts = {x: self.dice.count(x) for x in set(self.dice)}
            is_FH = 3 in counts.values() and 2 in counts.values()

            if is_FH:
                return sum(self.dice)  # Provided file logic used sum
            else:
                return 0

        elif category == "small_straight":
            uniq = sorted(list(set(self.dice)))
            if uniq == [1, 2, 3, 4, 5]:
                return 15
            else:
                return 0
        elif category == "large_straight":
            uniq = sorted(list(set(self.dice)))
            if uniq == [2, 3, 4, 5, 6]:
                return 20
            else:
                return 0
        elif category == "yahtzee":
            if len(set(self.dice)) == 1 and self.dice[0] != 0:
                return 50
            else:
                return 0
        elif category == "chance":
            return sum(self.dice)

    def get_upper_section_score(self):
        score = 0
        for category in ["ones", "twos", "threes", "fours", "fives", "sixes"]:
            value = self.categories.get(category)
            if value is not None:
                score += value
        return score

    def get_score(self):
        """Returns the score of the game"""
        score = 0
        for name, value in self.categories.items():
            if value is not None:
                score += value
        return score

    def select_category(self, category):
        """Select a category to place the dice in"""
        if self.categories[category] is None:
            self.categories[category] = self.calculate_score(category)
            self.categories_played += 1
            # Update bonus immediately
            self.categories["bonus"] = self.calculate_score("bonus")
        else:
            print("Category already used")

    def is_over(self):
        """Checks if the game is over"""
        return self.categories_played == 15
