from datetime import datetime, timedelta
import time
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from YahtzeeFast import YahtzeeFast
import argparse
import matplotlib.pyplot as plt

DICE_VALUES = 30
CATEGORIES = 16
UPPER_SECTION_FIELDS = 13
ROLLS_LEFT = 3

good_scores = [
    3,  # Ones
    6,  # Twos
    9,  # Threes
    12,  # Fours
    15,  # Fives
    18,  # Sixes
    10,  # Two of a kind
    18,  # Two pairs
    15,  # Three of a kind
    16,  # Four of a kind
    22,  # Full house
    15,  # Small straight
    20,  # Large straight
    50,  # Yahtzee
    22,  # Chance
]

max_scores = [
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
]


class YahtzeeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.dice_layer = nn.Sequential(
            nn.Linear(DICE_VALUES, 512),
            nn.ReLU(),
        )

        self.category_layer = nn.Sequential(
            nn.Linear(CATEGORIES + UPPER_SECTION_FIELDS + ROLLS_LEFT, 512),
            nn.ReLU(),
        )

        self.common_stream = nn.Sequential(
            nn.Linear(512 + 512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        dice_features = self.dice_layer(x[:, :DICE_VALUES])
        other_features = self.category_layer(x[:, DICE_VALUES:])

        combined_features = torch.cat((dice_features, other_features), dim=1)

        combined_features = self.common_stream(combined_features)

        values = self.value_stream(combined_features)
        advantages = self.advantage_stream(combined_features)

        # Q = V + (A - mean(A))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_vals


def train_model(policy_net, states, actions, targets, optimizer, clip_grad=1.0):
    policy_net.train()

    all_q_values = policy_net(states)

    q_pred = all_q_values.gather(1, actions)

    loss = F.smooth_l1_loss(q_pred, targets)

    optimizer.zero_grad()
    loss.backward()

    if clip_grad:
        nn.utils.clip_grad_norm_(policy_net.parameters(), clip_grad)

    optimizer.step()

    return loss.item()


def save_model(model: YahtzeeNet, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def self_play_fast(
    policy_net: YahtzeeNet,
    device,
    epsilon=0,
    plot_stats=False,
    games_to_play=8192,
):
    env = YahtzeeFast(games_to_play, device=device)
    steps_in_game = 3 * 15

    num_random_actions = int(steps_in_game * epsilon)
    actions = [True] * num_random_actions + [False] * (
        steps_in_game - num_random_actions
    )
    random.shuffle(actions)

    examples = []

    for is_random in actions:
        state_gpu = env.get_encoded_state()
        masks_gpu = env.get_action_masks()

        # Select Actions
        if is_random:
            noise = torch.rand(size=(games_to_play, masks_gpu.shape[1]), device=device)
            noise = noise.masked_fill(~masks_gpu, -1e12)
            actions_gpu = noise.argmax(dim=1)
        else:
            policy_net.eval()
            with torch.no_grad():
                q_vals = policy_net(state_gpu)
                probs = F.softmax(q_vals, dim=-1)
                probs = probs.masked_fill(~masks_gpu, 0.0)

                # if plot_stats:
                actions_gpu = probs.argmax(dim=1)
                # else:
                # actions_gpu = torch.multinomial(probs, 1).squeeze(1)

        # Step Environment
        rewards, dones, average_score = env.step(actions_gpu)

        next_state_gpu = env.get_encoded_state()
        next_masks_gpu = env.get_action_masks()

        batch_data = zip(
            state_gpu, actions_gpu, rewards, next_state_gpu, dones, next_masks_gpu
        )

        examples.extend(batch_data)

    if plot_stats:
        plot_statistics(examples, average_score, games_to_play)

    return examples, games_to_play, average_score


def plot_statistics(examples, total_average_score, games_played):
    category_rewards = [[0 for _ in range(max_scores[i] + 1)] for i in range(15)]
    roll_counts = [0] * 32
    bonus_count = 0
    yahtzee_count = 0
    for ex in examples:
        state, action, reward, next_state, done, next_mask = ex
        if action < 6 and reward > 50:
            category_rewards[action][int(reward) - 50] += 1
            bonus_count += 1
        elif action == 13 and reward == 50:
            category_rewards[action][int(reward)] += 1
            yahtzee_count += 1
        elif action < 15:
            category_rewards[action][int(reward)] += 1
        else:
            roll_counts[action - 15] += 1

    plt.figure(figsize=(25, 13))
    plt.suptitle(
        f"Games played: {games_played}, Average Score: {total_average_score:.2f}, Bonus Count: {bonus_count}",
        fontsize=16,
    )
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        if i < 15:
            plt.bar(
                range(len(category_rewards[i])),
                category_rewards[i],
                color="blue",
                alpha=0.7,
            )
            plt.title(f"Category {i} Rewards")
            plt.ylabel("Reward")
        else:
            plt.bar(range(32), roll_counts, color="green", alpha=0.7)
            plt.title("Roll Actions Count")
            plt.ylabel("Count")
    plt.show()


def train_from_scratch(
    policy_net: YahtzeeNet,
    target_net: YahtzeeNet,
    device,
    num_iterations=200,
    start_iter=0,
):
    """
    Train model from scratch using self-play.
    """
    print(f"Training on device: {device}")
    policy_net.to(device)
    target_net.to(device)
    TRAIN_COUNT = 500
    BATCH_SIZE = 1024
    best_avg_score = 0.0
    avg_score_list = []
    iteration_list = []
    epsilon_list = []

    all_examples = []
    period_start = time.perf_counter()
    estimated_time = None
    total_time_start = datetime.now()
    ten_last_times = []
    for iteration in range(start_iter, num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
        start_time = datetime.now()

        # epsilon = 0.9 - (iteration % 10 / 10)
        epsilon = 0.1

        examples, games_played, avg_score = self_play_fast(
            policy_net, device=device, epsilon=epsilon
        )
        all_examples.extend(examples)

        if avg_score > best_avg_score:
            best_avg_score = avg_score
        avg_score_list.append(avg_score)
        iteration_list.append(iteration + 1)
        epsilon_list.append(epsilon)

        play_time = datetime.now()
        print(
            f"Games played: {games_played}, Average score: {avg_score:.2f}, "
            f"Best score: {best_avg_score:.2f} "
            f"Epsilon: {epsilon:.3f}, Steps in memory: {len(all_examples)} "
            f"Play time: {(play_time - start_time).total_seconds():.2f}s"
        )

        # Train on collected data
        total_loss = 0.0
        optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
        for _ in range(TRAIN_COUNT):
            batch = random.sample(all_examples, BATCH_SIZE)
            states, actions, targets = calculate_targets(
                policy_net, target_net, batch, device
            )
            loss = train_model(policy_net, states, actions, targets, optimizer)
            total_loss += loss
        avg_loss = total_loss / TRAIN_COUNT
        print(
            f"Training loss: {avg_loss:.4f}, Training time: "
            f"{(datetime.now() - play_time).total_seconds():.2f}s"
        )
        # Remove 25% of old examples to keep memory size manageable
        if len(all_examples) > 8192 * 15 * 3:
            remove_count = len(all_examples) // 2
            all_examples = all_examples[remove_count:]

        current_time = time.perf_counter()
        if not ten_last_times:
            ten_last_times = [current_time - period_start] * 10
        else:
            ten_last_times[iteration % 10] = current_time - period_start
        period_start = current_time
        estimated_seconds = (
            sum(ten_last_times) / len(ten_last_times) * (num_iterations - iteration - 1)
        )
        estimated_time = timedelta(seconds=int(estimated_seconds))
        print(f"Estimated time remaining: {str(estimated_time).split('.')[0]}")

        # Update Target Network
        if (iteration + 1) % 2 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # Save checkpoint
        if (iteration + 1) % 20 == 0:
            save_model(policy_net, f"models/iter{iteration + 1}.pth")
            print(f"Model checkpoint saved at iteration {iteration + 1}")

    total_elapsed = datetime.now() - total_time_start
    plot_training_progress(
        iteration_list, avg_score_list, epsilon_list, best_avg_score, total_elapsed
    )
    return policy_net


def plot_training_progress(
    iterations, avg_scores, epsilons, best_avg_score, total_elapsed
):
    plt.figure(figsize=(10, 5))
    plt.title("Training completed")
    plt.suptitle(
        f"Best achieved average score: {best_avg_score:.2f}, Total training time: {str(total_elapsed).split('.')[0]}"
    )
    plt.plot(
        iterations, [avg_score.cpu() for avg_score in avg_scores], label="Average Score"
    )
    plt.plot(
        iterations, [e * 100 for e in epsilons], label="Random action probability (%)"
    )
    plt.title("Training Progress: Average Score over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average Score / Epsilon (%)")
    plt.grid(True)
    plt.legend()
    plt.show()


def calculate_targets(policy_net, target_net, batch, device, gamma=0.99):
    """
    Calculates the Bellman Targets using Double DQN logic.
    """

    states, actions, rewards, next_states, dones, next_masks = zip(*batch)
    # Stack and move to device
    states = torch.stack(states)
    actions = torch.stack(actions).unsqueeze(1)
    rewards = torch.stack(rewards).unsqueeze(1)
    next_states = torch.stack(next_states)
    dones = torch.stack(dones).unsqueeze(1)
    next_masks = torch.stack(next_masks)

    with torch.no_grad():  # Don't update gradients here
        next_q_values_raw = policy_net(next_states)

        next_q_values_raw = next_q_values_raw.masked_fill(~next_masks, -1e12)

        next_action_indices = next_q_values_raw.argmax(dim=1, keepdim=True)

        next_q_values = target_net(next_states).gather(1, next_action_indices)

        targets = rewards + (gamma * next_q_values * (1 - dones))

    return states, actions, targets


def benchmark_model(policy_net: YahtzeeNet, device):
    policy_net.to(device)
    policy_net.eval()
    self_play_fast(
        policy_net, device=device, epsilon=0.0, plot_stats=True, games_to_play=1000
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yahtzee AI Training and Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "benchmark"],
        help="Mode to run: train or benchmark",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to load/save the model",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=0,
        help="Starting iteration for training (used for loading checkpoints)",
    )
    args = parser.parse_args()
    input_size = DICE_VALUES + CATEGORIES + UPPER_SECTION_FIELDS + ROLLS_LEFT
    # Actions
    SELECT_CATEGORY = CATEGORIES - 1  # Bonus is not selectable
    ROLL_DICE = 32
    action_size = SELECT_CATEGORY + ROLL_DICE
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    policy_net = YahtzeeNet(input_size, action_size)
    target_net = YahtzeeNet(input_size, action_size)
    if args.model_path:
        policy_net.load_state_dict(torch.load(args.model_path))
        target_net.load_state_dict(policy_net.state_dict())

    if args.mode == "train":
        trained_model = train_from_scratch(
            policy_net,
            target_net,
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu",
            start_iter=args.start_iter,
        )
    if args.mode == "benchmark":
        benchmark_model(
            policy_net,
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu",
        )
