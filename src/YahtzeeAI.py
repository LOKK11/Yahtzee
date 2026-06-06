from datetime import datetime, timedelta
import time
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from YahtzeeFast import YahtzeeFast
from YahtzeeCategory import YahtzeeCategory
import argparse
import matplotlib.pyplot as plt
import torch.func as func

DICE_VALUES = 41
ROLLS_LEFT = 3
NORMALIZED_SCORE_YIELDS = 15
BONUS_YIELDS = 6

CATEGORIES_PLAYED = 16
NORMALIZED_UPPER_SECTION_SCORE = 1
NORMALIZED_CATEGORIES_PLAYED = 1
NORMALIZED_UPPER_CATEGORIES_PLAYED = 1
BONUS_AVAILABLE = 1

ROLL_SPECIFIC_INPUTS = DICE_VALUES + ROLLS_LEFT

COMMON_INPUTS = (
    NORMALIZED_SCORE_YIELDS
    + BONUS_YIELDS
    + CATEGORIES_PLAYED
    + NORMALIZED_UPPER_SECTION_SCORE
    + NORMALIZED_CATEGORIES_PLAYED
    + BONUS_AVAILABLE
)

CATEGORY_ACTIONS = 15
ROLL_ACTIONS = 211  # 211 possible hold patterns
NUM_POLICY_NETS = 3
SCORE_MULTIPLIER = 0.8

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

categories = [
    "Ones",
    "Twos",
    "Threes",
    "Fours",
    "Fives",
    "Sixes",
    "Two of a kind",
    "Two pairs",
    "Three of a kind",
    "Four of a kind",
    "Full house",
    "Small straight",
    "Large straight",
    "Yahtzee",
    "Chance",
]


class RollPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.common_stream = nn.Sequential(
            nn.Linear(
                ROLL_SPECIFIC_INPUTS + COMMON_INPUTS,
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
            nn.Linear(512, ROLL_ACTIONS),
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
                COMMON_INPUTS,
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
            nn.Linear(512, CATEGORY_ACTIONS),
        )

    def forward(self, x):
        common_features = self.common_stream(x)
        value = self.value_stream(common_features)
        advantages = self.advantage_stream(common_features)

        # Q = V + (A - mean(A))
        q_vals = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_vals


class BatchedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_nets):
        super().__init__()
        # Weights: [num_nets, in, out] | Biases: [num_nets, out]
        self.weights = nn.Parameter(torch.randn(num_nets, in_features, out_features))
        self.biases = nn.Parameter(torch.full((num_nets, out_features), 0.01))

        # Initialize weights properly for ELU by looping over each network
        for i in range(num_nets):
            nn.init.kaiming_uniform_(self.weights[i], a=1)

    def forward(self, x, indices):
        # w shape: [batch, in, out]
        w = self.weights[indices]
        b = self.biases[indices].unsqueeze(1)  # [batch, 1, out]

        # Batched Matrix Multiplication: [batch, 1, in] @ [batch, in, out]
        out = torch.bmm(x.unsqueeze(1), w) + b
        return out.squeeze(1)


class MultiCategoryNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = DICE_VALUES + ROLLS_LEFT
        num_nets = CATEGORY_ACTIONS

        # Common Stream
        self.common_fc = BatchedLinear(input_dim, 256, num_nets)

        # Value Stream
        self.value_fc1 = BatchedLinear(256, 256, num_nets)
        self.value_fc2 = BatchedLinear(256, 1, num_nets)
        # Advantage Stream
        self.adv_fc1 = BatchedLinear(256, 256, num_nets)
        self.adv_fc2 = BatchedLinear(256, ROLL_ACTIONS, num_nets)

    def forward(self, x, action_indices):
        """
        x: [batch_size, input_dim]
        action_indices: [batch_size] -> which category net to use for each game
        """
        # Common Features
        x = F.elu(self.common_fc(x, action_indices))

        # Value Stream
        v = F.elu(self.value_fc1(x, action_indices))
        v = self.value_fc2(v, action_indices)

        # Advantage Stream
        a = F.elu(self.adv_fc1(x, action_indices))
        a = self.adv_fc2(a, action_indices)

        # Q = V + (A - mean(A))
        q_vals = v + (a - a.mean(dim=1, keepdim=True))

        return q_vals


def train_model(
    policy_net,
    states,
    actions,
    targets,
    optimizer,
    clip_grad=1.0,
    entropy_multiplier=0.01,
):
    policy_net.train()

    q_values = policy_net(states)
    # 1. Q-Value loss
    q_value_preds = q_values.gather(1, actions)

    with torch.no_grad():
        surprise = torch.abs(targets - q_value_preds).detach()
        weight = 1.0 + (entropy_multiplier * surprise)

    raw_loss = F.smooth_l1_loss(q_value_preds, targets, reduction="none")
    weighted_loss = (weight * raw_loss).mean()

    # 2. Q-value backpropagation and update
    weighted_loss.backward()

    if clip_grad:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), clip_grad)

    optimizer.step()
    optimizer.zero_grad()

    return weighted_loss.item()


def get_batched_loss_fn(base_model, entropy_multiplier=0.01):
    """
    Creates a vectorized gradient function for N network instances.
    """

    def compute_single_loss(params, buffers, states, actions, targets):
        # Forward pass using functional_call (stateless)
        q_values = func.functional_call(base_model, (params, buffers), states)
        q_value_preds = q_values.gather(1, actions)

        # Surprise-weighted logic
        with torch.no_grad():
            surprise = torch.abs(targets - q_value_preds).detach()
            weight = 1.0 + (entropy_multiplier * surprise)

        raw_loss = F.smooth_l1_loss(q_value_preds, targets, reduction="none")
        weighted_loss = (weight * raw_loss).mean()
        return weighted_loss

    # Vectorize across dimension 0 (the N_INSTANCES dimension) for all inputs
    # This will compute gradients for all N models simultaneously
    return func.vmap(func.grad(compute_single_loss), in_dims=(0, 0, 0, 0, 0))

def prepare_vmap_ensemble(model, n_instances):
    """Stacks a list of models into a single vectorized function."""
    models_list = [model for _ in range(n_instances)]
    params, buffers = func.stack_module_state(models_list)

    def fcall(p, b, x):
        return func.functional_call(model, (p, b), (x,))

    vectorized_forward = func.vmap(fcall, in_dims=(0, 0, 0))
    
    return vectorized_forward, params, buffers


def update_vectorized_networks(
    optimizer, params, buffers, batched_grad_fn, states, actions, targets
):
    """Computes gradients and updates all N networks simultaneously."""
    
    batched_grads = batched_grad_fn(params, buffers, states, actions, targets)

    for key, param in params.items():
        param.grad = batched_grads[key]

    optimizer.step()
    optimizer.zero_grad()

def soft_update_ensemble(main_params, target_params, main_buffers, target_buffers, tau=0.005):
    """
    Applies soft updates to the target ensemble's parameters and buffers.
    """
    with torch.no_grad():
        for key in main_params.keys():
            target_params[key].mul_(1.0 - tau).add_(main_params[key], alpha=tau)
            
        for key in main_buffers.keys():
            if target_buffers[key].is_floating_point():
                target_buffers[key].mul_(1.0 - tau).add_(main_buffers[key], alpha=tau)
            else:
                target_buffers[key].copy_(main_buffers[key])


def train_ensemble(
    roll_policy_net: RollPolicyNet,
    category_policy_net: CategoryPolicyNet,
    roll_target_net: RollPolicyNet,
    category_target_net: CategoryPolicyNet,
    device,
    n_instances=8,
    num_iterations=1000,
    start_iter=0,
    pretrained_provided=False,
):
    print(f"Training {n_instances} instances simultaneously via vmap on {device}...")
    roll_policy_net.to(device)
    category_policy_net.to(device)
    roll_target_net.to(device)
    category_target_net.to(device)

    ROLL_BATCH_SIZE = 9
    CATEGORY_BATCH_SIZE = 1
    SIMUL_GAMES = 8192
    TRAINING_BATCH_SIZE = 2048
    TRAIN_COUNT = 64
    N_STEP = 2
    EVALUATION_FREQUENCY = 10

    roll_forward, roll_params, roll_buffers = prepare_vmap_ensemble(roll_policy_net, n_instances)
    category_forward, category_params, category_buffers = prepare_vmap_ensemble(category_policy_net, n_instances)
    roll_target_forward, roll_target_params, roll_target_buffers = prepare_vmap_ensemble(roll_target_net, n_instances)
    category_target_forward, category_target_params, category_target_buffers = prepare_vmap_ensemble(category_target_net, n_instances)

    roll_optimizer = optim.Adam(roll_params.values(), lr=1e-4)
    category_optimizer = optim.Adam(category_params.values(), lr=1e-4)
    roll_scheduler = torch.optim.lr_scheduler.ExponentialLR(roll_optimizer, gamma=1)
    category_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        category_optimizer, gamma=1
    )

    roll_examples = []
    category_examples = []
    current_examples = None

    gamma = 0.99

    for iteration in range(start_iter, num_iterations):
        print(f"\n=== Ensemble Iteration {iteration + 1}/{num_iterations} ===")
        currently_training = (
            "roll"
            if iteration % (ROLL_BATCH_SIZE + CATEGORY_BATCH_SIZE) < ROLL_BATCH_SIZE
            else "category"
        )
        if currently_training == "roll":
            policy_net = roll_policy_net
            forward = roll_forward
            params = roll_params
            buffers = roll_buffers
            target_forward = roll_target_forward
            target_params = roll_target_params
            target_buffers = roll_target_buffers
            optimizer = roll_optimizer
            scheduler = roll_scheduler
            examples_per_iteration = SIMUL_GAMES * 2 * 15
            current_examples = roll_examples
        else:
            policy_net = category_policy_net
            forward = category_forward
            params = category_params
            buffers = category_buffers
            target_forward = category_target_forward
            target_params = category_target_params
            target_buffers = category_target_buffers
            optimizer = category_optimizer
            scheduler = category_scheduler
            examples_per_iteration = SIMUL_GAMES * 15
            current_examples = category_examples

        epsilon = max(0.001, 1.0 - (iteration / 400))
        compute_batched_grads = get_batched_loss_fn(policy_net)
        examples, avg_score, _ = self_play_ensemble(
            roll_forward,
            category_forward,
            roll_params,
            category_params,
            roll_buffers,
            category_buffers,
            currently_training,
            device=device,
            epsilon=epsilon,
            games_to_play=SIMUL_GAMES,
            collect_examples=True,
            n_instances=n_instances,
            gamma=gamma,
        )

        current_examples.extend(examples)

        # Train on collected data
        for _ in range(TRAIN_COUNT):
            batch = random.sample(current_examples, TRAINING_BATCH_SIZE)
            states, actions, targets = calculate_targets_ensemble(
                target_forward,
                target_params,
                target_buffers,
                batch,
                TRAINING_BATCH_SIZE,
                gamma=gamma,
                n_step=N_STEP,
                n_instances=n_instances,
            )
            update_vectorized_networks(
                optimizer, params, buffers, compute_batched_grads, states, actions, targets
            )
        soft_update_ensemble(roll_params, roll_target_params, roll_buffers, roll_target_buffers, tau=0.01)


    # TODO: Continue here
    # 7. Final Evaluation and Selection
    print("\nTraining complete. Evaluating all instances...")
    best_score = -1
    best_idx = 0

    for i in range(n_instances):
        instance_weights = {k: v[i].detach() for k, v in roll_params.items()}
        roll_policy_net.load_state_dict(instance_weights)

        _, avg_score, _ = self_play(
            roll_policy_net,
            CategoryPolicyNet().to(device),
            "roll",
            device=device,
            epsilon=0.0,
            games_to_play=8192,
            collect_examples=False,
        )
        print(f"Final Evaluation Instance {i}: Score {avg_score:.2f}")
        if avg_score > best_score:
            best_score = avg_score
            best_idx = i

    print(f"\n=> Keeping Instance {best_idx} with Best Score: {best_score:.2f}")

    # Extract the winning weights and return standard model
    best_weights = {
        k: v[best_idx].detach().clone() for k, v in roll_params.items()
    }
    best_model = RollPolicyNet().to(device)
    best_model.load_state_dict(best_weights)

    return best_model


def save_model(model, path):
    """Save a single model to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def save_models(roll_policy_net, category_policy_net, path):
    """Save multiple networks to a single file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "roll_policy_net": roll_policy_net.state_dict(),
        "category_policy_net": category_policy_net.state_dict(),
    }

    torch.save(checkpoint, path)
    print(f"Models saved to {path}")


def load_models(roll_policy_net, category_policy_net, path, device="cpu"):
    """Load multiple networks from a single file."""
    if not os.path.isfile(path):
        print(f"Error: Model file '{path}' does not exist")
        return None

    checkpoint = torch.load(path, map_location=device)

    roll_policy_net.load_state_dict(checkpoint["roll_policy_net"])
    category_policy_net.load_state_dict(checkpoint["category_policy_net"])

    print(f"Models loaded from {path}")


def is_power_of_two(n: int) -> bool:
    return n > 0 and n.bit_count() == 1


def validate_parameters(games_to_play, n_step, n_sets):
    if not is_power_of_two(games_to_play):
        raise ValueError("Games to play must be a power of 2")
    if n_step < 1:
        raise ValueError("n_step must be at least 1")
    if not is_power_of_two(n_sets):
        raise ValueError("n_sets must be a power of 2")


def self_play(
    roll_policy_net,
    category_policy_net,
    currently_training,
    device,
    epsilon=0,
    games_to_play=8192,
    collect_stats=False,
    collect_examples=True,
    gamma=1.0,
    n_step=2,
    n_sets=8,
):
    env = YahtzeeFast(games_to_play, device=device)
    rounds_in_game = 15
    stats = None

    states_list = []
    actions_list = []
    rewards_list = []
    masks_list = []
    roll_action_stats_list = []
    category_actions_stats_list = []

    states_collected_per_round = (
        2 if currently_training == "roll" and not collect_stats else 1
    )

    for _ in range(rounds_in_game):
        for _ in range(2):
            state, actions, mask = select_roll_action(
                env, roll_policy_net, device, epsilon
            )
            if collect_examples and currently_training == "roll":
                states_list.append(state)
                actions_list.append(actions)
                masks_list.append(mask)
            if collect_stats:
                roll_action_stats_list.append(actions)

        state, actions, rewards, mask = select_category_action(
            env, category_policy_net, device, epsilon
        )

        if collect_examples and currently_training == "category":
            states_list.append(state)
            actions_list.append(actions)
            masks_list.append(mask)
        rewards_list.extend([rewards] * states_collected_per_round)
        if collect_stats:
            category_actions_stats_list.append(actions)

    all_rewards = torch.stack(rewards_list)
    if collect_stats:
        all_roll_action_stats = torch.stack(roll_action_stats_list)
        all_category_action_stats = torch.stack(category_actions_stats_list)
        stats = env.analyze_batch_stats(
            all_roll_action_stats.flatten(),
            all_category_action_stats.flatten(),
            all_rewards.flatten(),
        )
    average_score = env.get_average_final_score()
    if not collect_examples:
        return [], average_score, stats
    all_states = torch.stack(states_list)
    all_actions = torch.stack(actions_list)
    all_masks = torch.stack(masks_list)

    examples = []
    state_count = len(all_states)
    # Vectorized n-step calculation: Loop through time t and look ahead to t+n
    for t in range(state_count):
        rewards = torch.zeros(games_to_play, device=device)
        dones = torch.zeros(games_to_play, dtype=torch.float32, device=device)

        step_limit = n_step * states_collected_per_round

        for n in range(step_limit):
            action_idx = t + n
            if action_idx >= state_count:
                # If we look past the end of the game, flag as done
                dones = torch.ones(games_to_play, dtype=torch.float32, device=device)
                break

            discount = gamma ** (n // states_collected_per_round)
            rewards += all_rewards[action_idx] * discount

        # The state we bootstrap off of is `step_limit` steps in the future
        next_state_idx = min(t + step_limit, state_count - 1)

        batch_data = zip(
            all_states[t],
            all_actions[t],
            rewards,
            all_states[next_state_idx],
            dones,
            all_masks[next_state_idx],
        )
        examples.extend(batch_data)

    return examples, average_score, stats

def self_play_ensemble(
    roll_forward,
    category_forward,
    roll_params,
    category_params,
    roll_buffers,
    category_buffers,
    currently_training,
    device,
    epsilon=0,
    games_to_play=8192,
    collect_stats=False,
    collect_examples=True,
    gamma=1.0,
    n_step=2,
    n_instances=8,
):
    env = YahtzeeFast(games_to_play, device=device)
    rounds_in_game = 15
    stats = None

    states_list = []
    actions_list = []
    rewards_list = []
    masks_list = []
    roll_action_stats_list = []
    category_actions_stats_list = []

    states_collected_per_round = (
        2 if currently_training == "roll" and not collect_stats else 1
    )

    for _ in range(rounds_in_game):
        for _ in range(2):
            state, actions, mask = select_roll_action_ensemble(
                env, roll_forward, roll_params, roll_buffers, n_instances, device, epsilon, games_to_play,
            )
            if collect_examples and currently_training == "roll":
                states_list.append(state)
                actions_list.append(actions)
                masks_list.append(mask)
            if collect_stats:
                roll_action_stats_list.append(actions)

        state, actions, rewards, mask = select_category_action_ensemble(
            env, category_forward, category_params, category_buffers, n_instances, device, epsilon, games_to_play,
        )

        if collect_examples and currently_training == "category":
            states_list.append(state)
            actions_list.append(actions)
            masks_list.append(mask)
        rewards_list.extend([rewards] * states_collected_per_round)
        if collect_stats:
            category_actions_stats_list.append(actions)

    all_rewards = torch.stack(rewards_list)
    if collect_stats:
        all_roll_action_stats = torch.stack(roll_action_stats_list)
        all_category_action_stats = torch.stack(category_actions_stats_list)
        stats = env.analyze_batch_stats(
            all_roll_action_stats.flatten(),
            all_category_action_stats.flatten(),
            all_rewards.flatten(),
        )
    average_score = env.get_average_final_score()
    if not collect_examples:
        return [], average_score, stats
    all_states = torch.stack(states_list)
    all_actions = torch.stack(actions_list)
    all_masks = torch.stack(masks_list)

    examples = []
    state_count = len(all_states)
    # Vectorized n-step calculation: Loop through time t and look ahead to t+n
    for t in range(state_count):
        rewards = torch.zeros(games_to_play, device=device)
        dones = torch.zeros(games_to_play, dtype=torch.float32, device=device)

        step_limit = n_step * states_collected_per_round

        for n in range(step_limit):
            action_idx = t + n
            if action_idx >= state_count:
                # If we look past the end of the game, flag as done
                dones = torch.ones(games_to_play, dtype=torch.float32, device=device)
                break

            discount = gamma ** (n // states_collected_per_round)
            rewards += all_rewards[action_idx] * discount

        # The state we bootstrap off of is `step_limit` steps in the future
        next_state_idx = min(t + step_limit, state_count - 1)

        batch_data = zip(
            all_states[t],
            all_actions[t],
            rewards,
            all_states[next_state_idx],
            dones,
            all_masks[next_state_idx],
        )
        examples.extend(batch_data)

    return examples, average_score, stats

def select_roll_action(
    env: YahtzeeFast,
    roll_policy_net: RollPolicyNet,
    device,
    epsilon,
):
    state = env.get_encoded_state()
    roll_mask = env.get_roll_action_masks()

    roll_policy_net.eval()
    with torch.no_grad():
        q_values = roll_policy_net(state)
        q_values = q_values.masked_fill(~roll_mask, -1e12)

        roll = torch.rand(q_values.size(0), device=device)

        random_actions = (
            torch.rand_like(q_values).masked_fill(~roll_mask, -1e12).argmax(dim=1)
        )

        actions = torch.where(
            roll < epsilon,
            random_actions,
            q_values.argmax(dim=1),
        )

    _ = env.step(actions, action_type="roll")

    return state, actions, roll_mask

def select_roll_action_ensemble(
    env: YahtzeeFast, forward, params, buffers, n_instances, device, epsilon, games_to_play
):
    state = env.get_encoded_state()
    roll_mask = env.get_roll_action_masks()

    games_per_set = games_to_play // n_instances
    
    # Reshape from [games_to_play, state_dim] -> [n_instances, games_per_set, state_dim]
    state_reshaped = state.view(n_instances, games_per_set, -1)

    with torch.no_grad():
        # VMAP MAGIC: Passes n_instances batches through n_instances networks simultaneously
        q_values_reshaped = forward(params, buffers, state_reshaped)
        
        # Flatten back to [games_to_play, action_dim] for the environment
        q_values = q_values_reshaped.view(games_to_play, -1)
        
        q_values = q_values.masked_fill(~roll_mask, -1e12)
        roll = torch.rand(q_values.size(0), device=device)
        random_actions = (
            torch.rand_like(q_values).masked_fill(~roll_mask, -1e12).argmax(dim=1)
        )
        actions = torch.where(
            roll < epsilon, random_actions, q_values.argmax(dim=1)
        )

    _ = env.step(actions, action_type="roll")
    return state, actions, roll_mask


def select_category_action(
    env: YahtzeeFast, category_policy_net: CategoryPolicyNet, device, epsilon
):
    state = env.get_encoded_state()[:, ROLL_SPECIFIC_INPUTS:]
    mask = env.get_category_action_masks()
    category_policy_net.eval()
    with torch.no_grad():
        q_values = category_policy_net(state)
        q_values = q_values.masked_fill(~mask, -1e12)

        roll = torch.rand(q_values.size(0), device=device)

        random_actions = (
            torch.rand_like(q_values).masked_fill(~mask, -1e12).argmax(dim=1)
        )

        actions = torch.where(
            roll < epsilon,
            random_actions,
            q_values.argmax(dim=1),
        )

    rewards = env.step(actions, action_type="category")

    return state, actions, rewards, mask

def select_category_action_ensemble(
    env: YahtzeeFast, forward, params, buffers, n_instances, device, epsilon, games_to_play
):
    state = env.get_encoded_state()[:, ROLL_SPECIFIC_INPUTS:]
    mask = env.get_category_action_masks()
    
    games_per_instance = games_to_play // n_instances
    
    # Reshape from [games_to_play, state_dim] -> [n_instances, games_per_instance, state_dim]
    state_reshaped = state.view(n_instances, games_per_instance, -1)

    with torch.no_grad():
        q_values_reshaped = forward(params, buffers, state_reshaped)
        q_values = q_values_reshaped.view(games_to_play, -1)
        
        q_values = q_values.masked_fill(~mask, -1e12)
        roll = torch.rand(q_values.size(0), device=device)
        random_actions = (
            torch.rand_like(q_values).masked_fill(~mask, -1e12).argmax(dim=1)
        )
        actions = torch.where(
            roll < epsilon, random_actions, q_values.argmax(dim=1)
        )

    rewards = env.step(actions, action_type="category")
    return state, actions, rewards, mask

def update_yahtzee_stats(
    fig, rects, axes, stats, avg_score, games_played, epsilon, stop=False
):
    """Update the wide figure with new batch statistics."""
    fig.suptitle(
        f"Games: {games_played}, Avg Score: {avg_score:.2f}, "
        f"Bonuses: {stats['bonus_count']}, Yahtzees: {stats['yahtzee_count']} | "
        f"Epsilon: {epsilon:.4f}",
        fontsize=16,
    )

    for i in range(16):
        # Pull distribution: index 0-14 for categories, roll_counts for index 15
        dist = stats["cat_distributions"][i] if i < 15 else stats["roll_counts"]

        # Vectorized update of bar heights
        for rect, h in zip(rects[i], dist):
            rect.set_height(h.item())

        # Rescale the Y-axis so bars don't go off-screen
        axes[i].relim()
        axes[i].autoscale_view(scalex=False, scaley=True)

    if stop:
        plt.ioff()
        plt.show()
        return
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def setup_yahtzee_plots(interactive=True):
    """Initialize the figure with a wide bottom row for all of the hold patterns."""
    plt.style.use("dark_background")
    plt.ion()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(5, 4)

    axes = []
    for i in range(15):
        row = i // 4
        col = i % 4
        axes.append(fig.add_subplot(gs[row, col]))

    hold_ax = fig.add_subplot(gs[4, :])
    axes.append(hold_ax)

    rects = []
    for i in range(16):
        bar_count = ROLL_ACTIONS if i == 15 else 51

        color = "skyblue" if i < 15 else "salmon"
        rect = axes[i].bar(range(bar_count), [0] * bar_count, color=color, width=0.8)

        title = (
            f"{categories[i]}"
            if i < 15
            else f"Hold Patterns ({ROLL_ACTIONS} Quantity-Value Actions)"
        )
        axes[i].set_title(title, fontsize=10)

        axes[i].set_xticks([j for j in range(0, bar_count, 5)])

        rects.append(rect)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes, rects


def train_from_scratch(
    roll_policy_net: RollPolicyNet,
    category_policy_net: CategoryPolicyNet,
    roll_target_net: RollPolicyNet,
    category_target_net: CategoryPolicyNet,
    device,
    num_iterations=2000,
    start_iter=0,
):
    """
    Train model from scratch using self-play.
    """
    print(f"Training on device: {device}")
    roll_policy_net.to(device)
    category_policy_net.to(device)
    roll_target_net.to(device)
    category_target_net.to(device)
    BATCH_SIZE = 2048
    TRAIN_COUNT = 32
    SIMUL_GAMES = 4096
    EXAMPLE_BATCH_COUNT = 4
    EVALUATION_FREQUENCY = 10
    N_STEP = 2
    N_SETS = 8
    ROLL_BATCH_SIZE = 9
    CATEGORY_BATCH_SIZE = 1
    SCHEDULER_STEP_FREQUENCY = 10
    validate_parameters(SIMUL_GAMES, N_STEP, N_SETS)
    best_avg_score = 0.0
    avg_score_list = []
    iteration_list = []
    period_start = time.perf_counter()
    estimated_time = None
    total_time_start = datetime.now()
    previous_times = []
    fig, axes, rects = setup_yahtzee_plots()
    learning_rate = 1e-4
    roll_optimizer = optim.Adam(roll_policy_net.parameters(), lr=learning_rate)
    roll_scheduler = torch.optim.lr_scheduler.ExponentialLR(roll_optimizer, gamma=0.8)
    category_optimizer = optim.Adam(category_policy_net.parameters(), lr=learning_rate)
    category_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        category_optimizer, gamma=1
    )
    roll_examples = []
    category_examples = []
    current_examples = None
    try:
        for iteration in range(start_iter, num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            start_time = datetime.now()

            gamma = 1.0

            currently_training = (
                "roll"
                if iteration % (ROLL_BATCH_SIZE + CATEGORY_BATCH_SIZE) < ROLL_BATCH_SIZE
                else "category"
            )

            if currently_training == "roll":
                policy_net = roll_policy_net
                target_net = roll_target_net
                optimizer = roll_optimizer
                scheduler = roll_scheduler
                examples_per_iteration = SIMUL_GAMES * 2 * 15
                current_examples = roll_examples
            else:
                policy_net = category_policy_net
                target_net = category_target_net
                optimizer = category_optimizer
                scheduler = category_scheduler
                examples_per_iteration = SIMUL_GAMES * 15
                current_examples = category_examples

            if (iteration + 1) % EVALUATION_FREQUENCY == 0:
                epsilon = 0.0
                examples, avg_score, stats = self_play(
                    roll_policy_net,
                    category_policy_net,
                    currently_training,
                    device=device,
                    epsilon=epsilon,
                    games_to_play=SIMUL_GAMES,
                    collect_stats=True,
                    gamma=gamma,
                    n_step=N_STEP,
                    collect_examples=False,
                )
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    save_models(
                        roll_policy_net, category_policy_net, "models/best_model.pth"
                    )
                update_yahtzee_stats(
                    fig, rects, axes, stats, avg_score, SIMUL_GAMES, 0.0
                )
                avg_score_list.append(avg_score)
                iteration_list.append(iteration + 1)

            epsilon = max(0.001, 1.0 - (iteration / 400))
            examples, avg_score, _ = self_play(
                roll_policy_net,
                category_policy_net,
                currently_training,
                device=device,
                epsilon=epsilon,
                games_to_play=SIMUL_GAMES,
                collect_stats=False,
                gamma=gamma,
                n_step=N_STEP,
                collect_examples=True,
            )
            play_time = datetime.now()
            current_examples.extend(examples)
            print(
                f"Games played: {SIMUL_GAMES}, Average score: {avg_score:.2f}, "
                f"Best score: {best_avg_score:.2f} "
                f"Epsilon: {epsilon:.3f}, Currently training: {currently_training}, Steps in memory: {len(current_examples)} "
                f"Play time: {(play_time - start_time).total_seconds():.2f}s "
                f"Learning rate: {learning_rate:.6f}, Gamma: {gamma:.3f}"
            )

            # Train on collected data
            total_q_value_loss = 0.0
            for _ in range(TRAIN_COUNT):
                batch = random.sample(current_examples, BATCH_SIZE)
                states, actions, targets = calculate_targets(
                    target_net,
                    batch,
                    gamma=gamma,
                    n_step=N_STEP,
                )
                q_value_loss = train_model(
                    policy_net,
                    states,
                    actions,
                    targets,
                    optimizer,
                )
                total_q_value_loss += q_value_loss
            soft_update(target_net, policy_net, tau=0.01)

            avg_q_value_loss = total_q_value_loss / TRAIN_COUNT

            print(
                f"Training loss: {avg_q_value_loss:.4f}, Training time: "
                f"{(datetime.now() - play_time).total_seconds():.2f}s"
            )

            if len(current_examples) >= EXAMPLE_BATCH_COUNT * examples_per_iteration:
                del current_examples[:examples_per_iteration]

            if (iteration + 1) % SCHEDULER_STEP_FREQUENCY == 0:
                scheduler.step()
                learning_rate = scheduler.get_last_lr()[0]

            current_time = time.perf_counter()
            if not previous_times:
                previous_times = [current_time - period_start] * EVALUATION_FREQUENCY
            else:
                previous_times[iteration % EVALUATION_FREQUENCY] = (
                    current_time - period_start
                )
            period_start = current_time
            estimated_seconds = (
                sum(previous_times)
                / len(previous_times)
                * (num_iterations - iteration - 1)
            )
            estimated_time = timedelta(seconds=int(estimated_seconds))
            print(f"Estimated time remaining: {str(estimated_time).split('.')[0]}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        print("Saving current model...")
        save_models(
            roll_policy_net, category_policy_net, "models/interrupted_model.pth"
        )
    finally:
        total_elapsed = datetime.now() - total_time_start
        plot_training_progress(
            iteration_list, avg_score_list, best_avg_score, total_elapsed
        )
    return roll_policy_net


def play_category_rounds(
    multi_category_net,
    category,
    device,
    epsilon=0,
    games_to_play=4096,
):
    steps = 2
    rounds = 15
    second_best_chance = 0.01

    states_list, actions_list, rewards_list = [], [], []
    total_average_reward = 0.0

    for _ in range(rounds):
        env = YahtzeeCategory(games_to_play, category, device=device)
        for _ in range(steps):
            state = env.get_encoded_state()
            mask = env.get_action_masks(category)

            multi_category_net.eval()
            with torch.no_grad():
                q_values = multi_category_net(
                    state, torch.full((games_to_play,), category, device=device)
                )
                q_values = q_values.masked_fill(~mask, -1e12)

                roll = torch.rand(q_values.size(0), device=device)

                top_vals, top_indices = torch.topk(q_values, k=2, dim=1)
                best_actions = top_indices[:, 0]
                second_best_actions = torch.where(
                    top_vals[:, 1] > -1e11,
                    top_indices[:, 1],
                    best_actions,
                )

                random_actions = (
                    torch.rand_like(q_values).masked_fill(~mask, -1e12).argmax(dim=1)
                )

                actions = torch.where(
                    roll < epsilon,
                    random_actions,
                    torch.where(
                        roll < (epsilon + second_best_chance),
                        second_best_actions,
                        best_actions,
                    ),
                )

            env.step(actions)

            states_list.append(state)
            actions_list.append(actions)

        rewards, average_reward = env.step(actions)
        rewards_list.extend(
            [rewards / (max_scores[category] ** env.REWARD_EXPONENT)] * steps
        )
        total_average_reward += average_reward

    total_average_reward /= rounds
    all_states = torch.stack(states_list)
    all_actions = torch.stack(actions_list)
    all_rewards = torch.stack(rewards_list)

    examples = []
    for t in range(steps * rounds):
        batch_data = zip(
            all_states[t],
            all_actions[t],
            all_rewards[t],
        )
        examples.extend(batch_data)

    return examples, total_average_reward


def train_multi_category_net(
    multi_category_net: MultiCategoryNet,
    device,
    num_iterations=500,
    start_iter=0,
):
    """
    Train separate models for each category.
    """
    print(f"Training category models on device: {device}")
    multi_category_net.to(device)

    BATCH_SIZE = 2048
    TRAIN_COUNT = 32
    PLAY_COUNT = 2048

    optimizer = optim.Adam(multi_category_net.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    best_avg_scores = [0.0] * CATEGORY_ACTIONS

    try:
        for iteration in range(start_iter, num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            epsilon = max(0.001, 1.0 - (iteration / 25))
            for category in range(CATEGORY_ACTIONS):
                examples, avg_score = play_category_rounds(
                    multi_category_net,
                    category,
                    device=device,
                    epsilon=epsilon,
                    games_to_play=PLAY_COUNT,
                )

                if avg_score > best_avg_scores[category]:
                    best_avg_scores[category] = avg_score
                    save_model(multi_category_net, "models/multi_category_net_new.pth")

                total_loss = 0.0
                for _ in range(TRAIN_COUNT):
                    batch = random.sample(examples, BATCH_SIZE)
                    states, actions, rewards = zip(*batch)
                    states = torch.stack(states)
                    actions = torch.stack(actions).unsqueeze(1)
                    rewards = torch.stack(rewards).unsqueeze(1)
                    multi_category_net.train()
                    q_values = multi_category_net(
                        states, torch.full((BATCH_SIZE,), category, device=device)
                    )
                    q_value_preds = q_values.gather(1, actions)
                    q_value_loss = F.smooth_l1_loss(q_value_preds, rewards)
                    q_value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(multi_category_net.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += q_value_loss.item()

                print(
                    f"Games played: {PLAY_COUNT}, Category: {categories[category]} Average score: {avg_score:.2f}, "
                    f"Epsilon: {epsilon:.3f}, Steps in memory: {len(examples)}, Training loss: {total_loss / TRAIN_COUNT:.4f}"
                )
            scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted by user.")


def plot_training_progress(iterations, avg_scores, best_avg_score, total_elapsed):
    plt.ioff()
    plt.figure(figsize=(10, 5))
    plt.title("Training completed")
    plt.suptitle(
        f"Best achieved average score: {best_avg_score:.2f}, Total training time: {str(total_elapsed).split('.')[0]}"
    )
    plt.plot(iterations, [avg_score for avg_score in avg_scores], label="Average Score")
    plt.title("Training Progress: Average Score over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.legend()
    plt.show()


def soft_update(target_net, policy_net, tau):
    for target_param, policy_param in zip(
        target_net.parameters(), policy_net.parameters()
    ):
        target_param.data.copy_(
            tau * policy_param.data + (1.0 - tau) * target_param.data
        )


def calculate_targets(target_net, batch, gamma=1.0, n_step=2):
    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_masks,
    ) = zip(*batch)

    states = torch.stack(states)
    actions = torch.stack(actions).unsqueeze(1)
    rewards = torch.stack(rewards).unsqueeze(1)
    next_states = torch.stack(next_states)
    dones = torch.stack(dones).unsqueeze(1)
    next_masks = torch.stack(next_masks)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        # Mask out invalid future actions
        next_q_values = next_q_values.masked_fill(~next_masks, -1e12)

        # Bellman equation: Reward + Discounted Max Future Q
        # If the game is done, we don't add the future Q value.
        targets = rewards + next_q_values.max(dim=1, keepdim=True)[0] * (1 - dones) * (
            gamma**n_step
        )

    return states, actions, targets

def calculate_targets_ensemble(target_forward, target_params, target_buffers, batch, batch_size, gamma=1.0, n_step=2, n_instances=8):
    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_masks,
    ) = zip(*batch)

    states = torch.stack(states)
    actions = torch.stack(actions).unsqueeze(1)
    rewards = torch.stack(rewards).unsqueeze(1)
    next_states = torch.stack(next_states)
    dones = torch.stack(dones).unsqueeze(1)
    next_masks = torch.stack(next_masks)

    states_per_instance = batch_size // n_instances
    next_states = next_states.view(n_instances, states_per_instance, -1)

    with torch.no_grad():
        next_q_values = target_forward(target_params, target_buffers, next_states)
        next_q_values = next_q_values.view(batch_size, -1)
        # Mask out invalid future actions
        next_q_values = next_q_values.masked_fill(~next_masks, -1e12)

        # Bellman equation: Reward + Discounted Max Future Q
        # If the game is done, we don't add the future Q value.
        targets = rewards + next_q_values.max(dim=1, keepdim=True)[0] * (1 - dones) * (
            gamma**n_step
        )

    states = states.view(n_instances, states_per_instance, -1)
    actions = actions.view(n_instances, states_per_instance, -1)
    targets = targets.view(n_instances, states_per_instance, -1)

    return states, actions, targets


def benchmark_model(
    roll_policy_net: RollPolicyNet,
    device,
):
    SIMUL_GAMES = 8192
    roll_policy_net.to(device)
    _, average_score, stats = self_play(
        roll_policy_net,
        device=device,
        epsilon=0.0,
        games_to_play=8192,
        collect_stats=True,
    )
    fig, axes, rects = setup_yahtzee_plots()
    update_yahtzee_stats(
        fig, rects, axes, stats, average_score, SIMUL_GAMES, 0.0, stop=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yahtzee AI Training and Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "benchmark", "train_categories", "train_ensemble"],
        help="Mode to run: train or benchmark",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to load/save the model",
    )
    parser.add_argument(
        "--cat-model-path",
        type=str,
        default="",
        help="Path to load/save category model",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=0,
        help="Starting iteration for training (used for loading checkpoints)",
    )
    args = parser.parse_args()
    # Actions
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    roll_policy_net = RollPolicyNet()
    category_policy_net = CategoryPolicyNet()
    roll_target_net = RollPolicyNet()
    category_target_net = CategoryPolicyNet()
    multi_category_net = MultiCategoryNet()
    if args.model_path:
        load_models(
            roll_policy_net, category_policy_net, args.model_path, device=device
        )
        roll_target_net.load_state_dict(roll_policy_net.state_dict())
        category_target_net.load_state_dict(category_policy_net.state_dict())

    if args.cat_model_path:
        multi_category_net.load_state_dict(torch.load(args.cat_model_path))

    if args.mode == "train":
        trained_model = train_from_scratch(
            roll_policy_net,
            category_policy_net,
            roll_target_net,
            category_target_net,
            device=device,
            start_iter=args.start_iter,
        )

    if args.mode == "benchmark":
        benchmark_model(
            roll_policy_net,
            device,
        )

    if args.mode == "train_categories":
        train_multi_category_net(
            multi_category_net,
            device,
            num_iterations=500,
            start_iter=args.start_iter,
        )

    if args.mode == "train_ensemble":
        trained_model = train_ensemble(
            roll_policy_net,
            category_policy_net,
            roll_target_net,
            category_target_net,
            device=device,
            n_instances=8,
            num_iterations=50,
        )
        save_models(
            trained_model, category_policy_net, "models/best_ensemble_model.pth"
        )
