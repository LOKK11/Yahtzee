import random
import torch
import numpy as np
from YahtzeeAI import YahtzeeNet

MODEL = "models/avg200.pth"


class Yahtzee:
    def __init__(self):
        self.dice = [0] * 5
        self.locked = []
        self.rolls_left = 3
        # Use a list for categories to maintain index order 0-15 matching the AI
        self.category_names = [
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
        self.categories = {name: None for name in self.category_names}
        self.categories_played = 0
        self.neural_network = False
        self.model = None
        self.device = torch.device("cpu")

        # Roll Action Masks (copied from YahtzeeFast for decoding AI moves)
        self.roll_action_masks = np.zeros((32, 5), dtype=bool)
        for i in range(32):
            for bit in range(5):
                if (i >> bit) & 1:
                    self.roll_action_masks[i, bit] = True

    def load_model(self):
        """Loads the latest model from the models directory."""
        try:
            input_dim = (
                62  # 30 (dice) + 16 (categories) + 13 (upper score) + 3 (rolls left)
            )
            output_dim = 47  # 15 categories + 32 roll combinations

            self.model = YahtzeeNet(input_dim, output_dim)
            self.model.load_state_dict(torch.load(MODEL, map_location=self.device))
            self.model.eval()
            self.neural_network = True
            print(f"Loaded model: {MODEL}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def get_encoded_state(self):
        """Encodes the current state into a tensor (1, 62) for the AI."""
        encoded = np.zeros((1, 62), dtype=np.float32)

        current_dice = np.array(self.dice)
        if np.sum(current_dice) > 0:
            val_offsets = (current_dice - 1) * 5
            pos_indices = np.arange(5)
            flat_indices = val_offsets + pos_indices
            encoded[0, flat_indices] = 1.0

        for i, name in enumerate(self.category_names):
            if self.categories[name] is not None:
                encoded[0, 30 + i] = 1.0

        # 3. Upper Section Score (13 inputs)
        # Calculate upper score
        upper_score = 0
        for name in self.category_names[:6]:
            val = self.categories[name]
            if val is not None:
                upper_score += val

        upper_score = min(upper_score, 63)
        upper_bucket = int(upper_score // (63 / 12))
        encoded[0, 46 + upper_bucket] = 1.0

        encoded[0, 59 + self.rolls_left] = 1.0

        return torch.tensor(encoded, dtype=torch.float32).to(self.device)

    def get_action_mask(self):
        """Returns boolean mask for valid actions."""
        mask = torch.zeros((1, 47), dtype=torch.bool)

        can_pick_category = self.rolls_left == 0

        if not can_pick_category:
            mask[0, 15:] = True
        else:
            for i in range(15):
                if self.categories[self.category_names[i]] is None:
                    mask[0, i] = True

        return mask

    def get_ai_prediction(self):
        """Returns the best action string description."""
        if not self.model or not self.neural_network:
            return ""

        state = self.get_encoded_state()
        mask = self.get_action_mask()

        with torch.no_grad():
            q_values = self.model(state)
            q_values = q_values.masked_fill(~mask, -1e12)
            action_idx = q_values.argmax().item()

        # Decode Action
        if action_idx < 15:
            cat_name = self.category_names[action_idx].replace("_", " ").title()
            return f"Pick Category: {cat_name}"
        else:
            roll_mask_idx = action_idx - 15
            keep_mask = self.roll_action_masks[roll_mask_idx]

            keep_indices = [i + 1 for i, keep in enumerate(keep_mask) if keep]

            if not keep_indices:
                return "Roll Again (Keep None)"
            else:
                return f"Roll Again (Keep positions: {keep_indices})"

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
