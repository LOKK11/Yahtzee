import random


class Yahtzee:
    def __init__(self):
        self.dice = []
        self.locked = []
        self.locks_used = 0
        self.rolls_left = 3
        self.categories = {
            "ones": None,
            "twos": None,
            "threes": None,
            "fours": None,
            "fives": None,
            "sixes": None,
            "two_of_a_kind": None,
            "two_pairs": None,
            "three_of_a_kind": None,
            "four_of_a_kind": None,
            "full_house": None,
            "small_straight": None,
            "large_straight": None,
            "yahtzee": None,
            "chance": None,
            "bonus": None,
        }

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
        self.dice = []
        self.locked = []
        self.rolls_left = 3
        self.locks_used = 0

    def lock_dice(self, dice_to_lock):
        """Locks the dice that the player wants to keep"""
        self.locked.append(dice_to_lock)
        self.locks_used += 1

    def lock_dice_with_number(self, number):
        """Locks one dice with the specified number"""
        for i in range(5):
            if self.dice[i] == number and i not in self.locked:
                self.locked.append(i)
                self.locks_used += 1
                break

    def unlock_dice(self, dice_to_unlock):
        """Unlocks the dice that the player wants to unlock"""
        self.locked.remove(dice_to_unlock)
        self.locks_used += 1

    def unlock_dice_with_number(self, number):
        """Unlocks one dice with the specified number"""
        for i in range(5):
            if self.dice[i] == number and i in self.locked:
                self.locked.remove(i)
                self.locks_used += 1
                break

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
            if sum(self.dice) >= 63:
                return 50
            else:
                return 0

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
            three_of_a_kind = None
            two_of_a_kind = None
            for i in range(6, 0, -1):
                if self.dice.count(i) >= 3:
                    three_of_a_kind = i
                    continue
                if self.dice.count(i) >= 2:
                    two_of_a_kind = i
            if three_of_a_kind and two_of_a_kind:
                return three_of_a_kind * 3 + two_of_a_kind * 2
            else:
                return 0
        elif category == "small_straight":
            if sorted(self.dice) == [1, 2, 3, 4, 5]:
                return 15
            else:
                return 0
        elif category == "large_straight":
            if sorted(self.dice) == [2, 3, 4, 5, 6]:
                return 20
            else:
                return 0
        elif category == "yahtzee":
            if self.dice.count(self.dice[0]) == 5:
                return 50
            else:
                return 0
        elif category == "chance":
            return sum(self.dice)

    def get_normalized_score(self, category):
        """Returns the normalized score for a category"""
        max_scores = {
            "ones": 5,
            "twos": 10,
            "threes": 15,
            "fours": 20,
            "fives": 25,
            "sixes": 30,
            "bonus": 50,
            "two_of_a_kind": 12,
            "two_pairs": 22,
            "three_of_a_kind": 18,
            "four_of_a_kind": 24,
            "full_house": 25,
            "small_straight": 15,
            "large_straight": 20,
            "yahtzee": 50,
            "chance": 30,
        }
        score = self.categories.get(category)
        if score is None:
            return 0.0
        max_score = max_scores.get(category)
        return score / max_score

    def get_upper_section_score(self):
        """Returns the score of the upper section"""
        score = 0
        for category in ["ones", "twos", "threes", "fours", "fives", "sixes"]:
            value = self.categories.get(category)
            if value is not None:
                score += value
        return score

    def get_total_score(self):
        """Returns the score of the game"""
        score = 0
        for _, value in self.categories.items():
            if value is not None:
                score += value
        return score

    def get_normalized_total_score(self):
        """Returns the normalized total score"""
        total_score = self.get_total_score()
        max_total_score = 371  # Maximum possible score in Yahtzee
        return total_score / max_total_score

    def select_category(self, category):
        """Select a category to place the dice in"""
        if self.categories[category] is None:
            self.categories[category] = self.calculate_score(category)
        else:
            print("Category already used")

    def get_valid_categories(self):
        """Returns a list of valid categories"""
        return [
            cat
            for cat, val in self.categories.items()
            if val is None and cat != "bonus"
        ]

    def get_valid_actions(self):
        """Returns a list of valid actions"""
        CATEGORIES = 15
        DICE_VALUES = 6
        actions = []
        if self.rolls_left < 3:
            for cat in self.get_valid_categories():
                actions.append(
                    list(self.categories.keys()).index(cat)
                )  # Category action
        if self.rolls_left > 0:
            actions.append(CATEGORIES)  # Roll action
        if self.locks_used < 5:
            locked_values = [self.dice[i] for i in self.locked]
            for i in range(1, DICE_VALUES + 1):
                if i in self.dice and i not in locked_values:
                    actions.append(CATEGORIES + i)  # Lock action
                if i in locked_values:
                    actions.append(CATEGORIES + DICE_VALUES + i)  # Unlock action

        return actions

    def get_action_mask(self):
        """Returns a boolean mask of valid actions."""
        mask = [False] * 28  # Initialize all actions as invalid
        valid_actions = self.get_valid_actions()
        for action in valid_actions:
            mask[action] = True
        return mask

    def is_over(self):
        """Checks if the game is over"""
        return all(
            value is not None
            for key, value in self.categories.items()
            if key != "bonus"
        )

    def take_action(self, action):
        """Takes an action in the game"""
        CATEGORIES = 15
        DICE_VALUES = 6

        if action < CATEGORIES:
            category_list = list(self.categories.keys())
            category = category_list[action]
            self.select_category(category)
            self.reset_dice()
        elif action == CATEGORIES:
            self.roll_dice()
        elif CATEGORIES < action <= CATEGORIES + DICE_VALUES:
            number_to_lock = action - CATEGORIES
            self.lock_dice_with_number(number_to_lock)
        else:
            number_to_unlock = action - CATEGORIES - DICE_VALUES
            self.unlock_dice_with_number(number_to_unlock)

    def get_normalized_dice_counts(self):
        """Returns a list with the normalized counts of each dice value"""
        counts = [0] * 6
        for die in self.dice:
            counts[die - 1] += 1 / 5.0  # Normalize counts
        return counts

    def get_normalized_locked_counts(self):
        """Returns a list with the normalized counts of locked dice values"""
        counts = [0] * 6
        for i in self.locked:
            die = self.dice[i]
            counts[die - 1] += 1 / 5.0  # Normalize counts
        return counts
