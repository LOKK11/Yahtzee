import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from Yahtzee import Yahtzee


# Simple feed-forward network with policy and value heads for Yahtzee decisions.
class YahtzeeNet(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)  # logits for actions
        self.value_head = nn.Linear(hidden_size, 1)  # state value

    def forward(self, x):
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


# Convert a game state into a flat tensor.
def encode_state(game: Yahtzee) -> torch.Tensor:
    counts = game.get_normalized_dice_counts()

    locked = game.get_normalized_locked_counts()

    sc_vec = []
    for cat in game.categories.keys():
        sc_vec.append(game.get_normalized_score(cat))

    # rolls_left scalar
    rolls_left_norm = [float(game.rolls_left) / 3.0]

    # Total score scalar
    total_score_norm = [game.get_normalized_total_score()]
    return torch.tensor(
        counts + locked + sc_vec + rolls_left_norm + total_score_norm,
        dtype=torch.float32,
    )


# Dataset expects tuples of (state_tensor, action_index, return_value, advantage)
# For supervised / imitation you can set advantage = 1 and return_value = target value
class YahtzeeDataset(Dataset):
    def __init__(self, examples):
        # examples: list of dicts with keys: 'state', 'action', 'value', 'adv'
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["state"], ex["action"], ex["value"], ex.get("adv", 1.0)


# Train function using policy gradient style loss + value MSE
def train_model(
    model, dataset, epochs=10, batch_size=64, lr=1e-4, clip_grad=1.0, device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        for states, actions, values, advs in loader:
            states = states.to(device)
            actions = actions.to(device)
            values = values.to(device)
            advs = advs.to(device)

            logits, preds = model(states)
            logp = F.softmax(logits, dim=-1)
            # Gather log-prob of taken actions
            logp_a = logp.gather(1, actions.unsqueeze(1)).squeeze(1)

            policy_loss = -(logp_a * advs).mean()
            value_loss = F.mse_loss(preds, values)

            # entropy bonus to encourage exploration
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * logp).sum(dim=-1).mean()
            loss = policy_loss + value_loss - 0.01 * entropy

            opt.zero_grad()
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

            epoch_policy_loss += policy_loss.item() * states.size(0)
            epoch_value_loss += value_loss.item() * states.size(0)

        n = len(dataset)
        # simple progress print; remove or adapt to logging in your app
        print(
            f"Epoch {epoch + 1}/{epochs} policy_loss={epoch_policy_loss / n:.4f} value_loss={epoch_value_loss / n:.4f}"
        )

    return model


# Choose action given a raw game state. Returns (action_index, probs_tensor)
def select_action(
    model,
    game,
    action_mask=None,
    deterministic=False,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = encode_state(game).unsqueeze(0).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(state)
        if action_mask is not None:
            mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
            large_neg = -1e9
            logits = logits.masked_fill(~mask.unsqueeze(0), large_neg)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        if deterministic:
            action = int(probs.argmax().item())
        else:
            action = int(torch.multinomial(probs, 1).item())
    return action, probs


# Save / load helpers
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path, input_size, hidden_size, action_size, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YahtzeeNet(input_size, hidden_size, action_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def self_play_game(model, epsilon=0.1, device=None):
    """
    Play one game using the model (with epsilon-greedy exploration).
    Returns list of examples with (state, action, reward, advantage).
    """
    states = []
    actions = []
    rewards = []

    game = Yahtzee()

    while not game.is_over():
        state = encode_state(game)

        # Epsilon-greedy: random action with probability epsilon
        if random.random() < epsilon:
            valid_actions = game.get_valid_actions()
            action = random.choice(valid_actions)
        else:
            action_mask = game.get_action_mask()  # Boolean mask of valid actions
            action, _ = select_action(
                model, game, action_mask, deterministic=False, device=device
            )

        old_score = game.get_normalized_total_score()
        game.take_action(action)
        new_score = game.get_normalized_total_score()
        # print(f"Action taken: {action}, Score: {new_score}")

        reward = new_score - old_score

        states.append(state)
        actions.append(torch.tensor(action, dtype=torch.long))
        rewards.append(reward)

    # Calculate returns
    # returns = calculate_returns(rewards, gamma=0.99)

    # Build examples
    examples = []
    total_score = game.get_normalized_total_score()
    for i in range(len(states)):
        examples.append(
            {
                "state": states[i],
                "action": actions[i],
                "value": torch.tensor(
                    (rewards[i] + total_score) / 2, dtype=torch.float32
                ),
                "adv": torch.tensor(1.0, dtype=torch.float32),
            }
        )

    return examples, game.get_total_score()


def calculate_returns(rewards, gamma=0.99):
    """Calculate discounted returns."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_from_scratch(
    model: YahtzeeNet, num_iterations=1000, games_per_iter=50, device=None
):
    """
    Train model from scratch using self-play.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # Collect games
        all_examples = []
        scores = []
        epsilon = max(
            0.1, 1.0 - iteration / (num_iterations * 0.5)
        )  # Decay exploration

        for _ in range(games_per_iter):
            examples, final_score = self_play_game(
                model, epsilon=epsilon, device=device
            )
            all_examples.extend(examples)
            scores.append(final_score)

        avg_score = sum(scores) / len(scores)
        print(
            f"Games played: {games_per_iter}, Avg score: {avg_score:.1f}, Epsilon: {epsilon:.3f}"
        )

        # Improve advantages using baseline
        if iteration > 0:  # After first iteration, use value predictions as baseline
            dataset_temp = YahtzeeDataset(all_examples)
            loader = DataLoader(
                dataset_temp, batch_size=len(all_examples), shuffle=False
            )

            model.eval()
            with torch.no_grad():
                for states, actions, values, _ in loader:
                    states = states.to(device)
                    values = values.to(device)
                    _, baseline_values = model(states)
                    advantages = values - baseline_values

                    # Update advantages in examples
                    for i, ex in enumerate(all_examples):
                        ex["adv"] = advantages[i]

        # Train on collected data
        dataset = YahtzeeDataset(all_examples)
        train_model(model, dataset, epochs=5, batch_size=64, lr=1e-3, device=device)

        # Save checkpoint
        if (iteration + 1) % 100 == 0:
            save_model(model, f"models/yahtzee_iter_{iteration + 1}.pth")

    return model


# Example minimal usage (replace dataset building with your self-play / recorded games).
if __name__ == "__main__":
    DICE_VALUES = 6
    LOCKED_DICES = 6
    ROLLS_LEFT = 1
    CATEGORIES = 16
    TOTAL_SCORE = 1
    input_size = DICE_VALUES + LOCKED_DICES + ROLLS_LEFT + CATEGORIES + TOTAL_SCORE
    # Actions
    SELECT_CATEGORY = CATEGORIES - 1  # Bonus is not selectable
    ROLL_DICE = 1
    LOCK_DICE = 6
    UNLOCK_DICE = 6
    action_size = SELECT_CATEGORY + ROLL_DICE + LOCK_DICE + UNLOCK_DICE
    model = YahtzeeNet(input_size, 128, action_size)

    # Train from scratch
    trained_model = train_from_scratch(
        model,
        num_iterations=100,
        games_per_iter=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Save example
    save_model(model, "/tmp/yahtzee_net.pth")
