import argparse
import torch
import os
import matplotlib.pyplot as plt
from networks import RollPolicyNet, CategoryPolicyNet
import config as cfg
from YahtzeeFast import YahtzeeFast


class TrainEnsemble:
    def __init__(self, device, args):
        self.device = device
        self.model_path = args.model_path
        self.start_iter = args.start_iter
        self.n_instances = args.n_instances
        self.num_iterations = args.num_iterations
        self.parallel_games = args.parallel_games
        self.training_batch_size = args.training_batch_size
        self.learning_rate = args.learning_rate
        self.learning_rate_gamma = args.learning_rate_gamma
        self.training_loops = args.training_loops
        self.roll_cat_ratio = args.roll_cat_ratio
        self.epsilon_gamma = args.epsilon_gamma
        self.start_epsilon = args.start_epsilon
        self.min_epsilon = args.min_epsilon
        self.n_step = args.n_step
        self.evaluation_frequency = args.evaluation_frequency
        self.max_roll_states = args.max_roll_states
        self.max_cat_states = args.max_category_states
        self.gamma = args.gamma

        self.evaluating = False
        self.games_per_instance = self.parallel_games // self.n_instances
        self.max_roll_states_per_instance = self.max_roll_states // self.n_instances
        self.max_cat_states_per_instance = self.max_cat_states // self.n_instances
        self.roll_mem_pointer = 0
        self.cat_memory_pointer = 0
        self.roll_states_in_memory = 0
        self.cat_states_in_memory = 0

        self.setup_networks()

        self.setup_plots()

        self.setup_vmaps()

        self.setup_optimizers()

        self.setup_schedulers()

        self.env = YahtzeeFast(self.parallel_games, self.device)

        self.setup_state_buffers()

        self.best_score = 0

    def load_models(self, model_path):
        """Load multiple networks from a single file."""
        if not os.path.isfile(model_path):
            print(f"Error: Model file '{model_path}' does not exist")
            return None

        checkpoint = torch.load(model_path, map_location=self.device)

        self.roll_policy_net.load_state_dict(checkpoint["roll_policy_net"])
        self.cat_policy_net.load_state_dict(checkpoint["category_policy_net"])

        print(f"Models loaded from {model_path}")

    def setup_networks(self):
        self.roll_policy_net = RollPolicyNet().to(self.device)
        self.cat_policy_net = CategoryPolicyNet().to(self.device)
        self.target_roll_net = RollPolicyNet().to(self.device)
        self.target_cat_net = CategoryPolicyNet().to(self.device)
        if self.model_path:
            self.load_models(self.model_path)

    def setup_plots(self):
        """Initialize the plot view for the training loop"""
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
            bar_count = cfg.ROLL_ACTIONS if i == 15 else 51

            color = "skyblue" if i < 15 else "salmon"
            rect = axes[i].bar(range(bar_count), [0] * bar_count, color=color, width=0.8)

            title = (
                f"{cfg.CATREGORY_NAMES[i]}"
                if i < 15
                else f"Hold Patterns ({cfg.ROLL_ACTIONS} Quantity-Value Actions)"
            )
            axes[i].set_title(title, fontsize=10)

            axes[i].set_xticks([j for j in range(0, bar_count, 5)])

            rects.append(rect)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig = fig
        self.axes = axes
        self.rects = rects

    def prepare_vmap_ensemble(self, model):
        """Stacks a list of models into a single vectorized function."""
        models_list = [model for _ in range(self.n_instances)]
        params, buffers = torch.func.stack_module_state(models_list)

        def fcall(p, b, x):
            return torch.func.functional_call(model, (p, b), (x,))

        vectorized_forward = torch.func.vmap(fcall, in_dims=(0, 0, 0))

        return vectorized_forward, params, buffers

    def setup_vmaps(self):
        self.roll_forward, self.roll_params, self.roll_buffers = self.prepare_vmap_ensemble(
            self.roll_policy_net
        )
        self.cat_forward, self.cat_params, self.cat_buffers = self.prepare_vmap_ensemble(
            self.cat_policy_net
        )
        self.roll_target_forward, self.roll_target_params, self.roll_target_buffers = (
            self.prepare_vmap_ensemble(self.target_roll_net)
        )
        self.cat_target_forward, self.cat_target_params, self.cat_target_buffers = (
            self.prepare_vmap_ensemble(self.target_cat_net)
        )

    def setup_optimizers(self):
        self.roll_optimizer = torch.optim.Adam(self.roll_params.values(), lr=self.learning_rate)
        self.cat_optimizer = torch.optim.Adam(self.cat_params.values(), lr=self.learning_rate)

    def setup_schedulers(self):
        self.roll_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.roll_optimizer, gamma=self.learning_rate_gamma
        )
        self.cat_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.cat_optimizer, gamma=self.learning_rate_gamma
        )

    def setup_state_buffers(self):
        roll_state_shape = (
            self.n_instances,
            self.max_roll_states_per_instance,
            self.env.ENCODED_STATE_SIZE,
        )
        roll_action_shape = (
            self.n_instances,
            self.max_roll_states_per_instance,
            1,
        )

        self.roll_states = torch.zeros(roll_state_shape, device=self.device, dtype=torch.float)
        self.roll_actions = torch.zeros(roll_action_shape, device=self.device, dtype=torch.uint16)
        self.roll_rewards = torch.zeros(roll_action_shape, device=self.device, dtype=torch.uint16)
        self.roll_next_states = torch.zeros(roll_state_shape, device=self.device, dtype=torch.float)
        self.roll_dones = torch.zeros(roll_action_shape, device=self.device, dtype=torch.bool)
        self.roll_next_masks = torch.zeros(roll_state_shape, device=self.device, dtype=torch.bool)

        cat_state_shape = (
            self.n_instances,
            self.max_cat_states_per_instance,
            self.env.ENCODED_STATE_SIZE,
        )
        cat_action_shape = (
            self.n_instances,
            self.max_cat_states_per_instance,
            1,
        )

        self.cat_states = torch.zeros(cat_state_shape, device=self.device, dtype=torch.float)
        self.cat_actions = torch.zeros(cat_action_shape, device=self.device, dtype=torch.int32)
        self.cat_rewards = torch.zeros(cat_action_shape, device=self.device, dtype=torch.int32)
        self.cat_next_states = torch.zeros(cat_state_shape, device=self.device, dtype=torch.float)
        self.cat_dones = torch.zeros(cat_action_shape, device=self.device, dtype=torch.bool)
        self.cat_next_masks = torch.zeros(cat_state_shape, device=self.device, dtype=torch.bool)

    def train(self):
        """Train the RollPolicyNet and CategoryPolicyNet using n parallel
        instances of the models so that on every x instances the best
        model is chosen and the rest of the models are updated with the
        state of the best model."""

        print(f"Training {self.n_instances} on device: {self.device}")
        for iteration in range(self.start_iter, self.num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.num_iterations} ===")
            self.epsilon = max(self.min_epsilon, self.epsilon_gamma**iteration)
            self.currently_training = (
                cfg.ROLL
                if iteration % (self.roll_cat_ratio[0] + self.roll_cat_ratio[1])
                < self.roll_cat_ratio[0]
                else cfg.CATEGORY
            )
            if self.currently_training == cfg.ROLL:
                self.roll_iteration()
            else:
                self.category_iteration()

    def get_batched_loss_fn(self, base_model, entropy_multiplier=0.01):
        """
        Creates a vectorized gradient function for N network instances.
        """

        def compute_single_loss(params, buffers, states, actions, targets):
            # Forward pass using functional_call (stateless)
            q_values = torch.func.functional_call(base_model, (params, buffers), states)
            q_value_preds = q_values.gather(1, actions)

            # Surprise-weighted logic
            with torch.no_grad():
                surprise = torch.abs(targets - q_value_preds).detach()
                weight = 1.0 + (entropy_multiplier * surprise)

            raw_loss = torch.nn.functional.smooth_l1_loss(q_value_preds, targets, reduction="none")
            weighted_loss = (weight * raw_loss).mean()
            return weighted_loss

        # Vectorize across dimension 0 (the N_INSTANCES dimension) for all inputs
        # This will compute gradients for all N models simultaneously
        return torch.func.vmap(torch.func.grad(compute_single_loss), in_dims=(0, 0, 0, 0, 0))

    def roll_iteration(self):
        """Do one training iteration for roll category network."""
        self.play_with_roll_net()
        compute_grads = self.get_batched_loss_fn(self.roll_policy_net)
        self.train_roll_net()

    def category_iteration(self):
        pass

    def select_roll_action(self, epsilon):
        # Reshape from [games_to_play, state_dim] -> [n_instances, games_per_instance, state_dim]
        state = self.env.get_encoded_state()
        mask = self.env.get_roll_action_masks()
        state_reshaped = state.view(self.n_instances, self.games_per_instance, -1)

        with torch.no_grad():
            # Passes n_instances batches through n_instances networks simultaneously
            q_values_reshaped = self.roll_forward(
                self.roll_params, self.roll_buffers, state_reshaped
            )
            q_values = q_values_reshaped.view(self.parallel_games, -1)

            q_values = q_values.masked_fill(~mask, -1e12)
            roll = torch.rand(q_values.size(), device=self.device)
            random_actions = torch.rand_like(q_values).masked_fill(~mask, -1e12).argmax(dim=1)
            # TODO: Fix bug here
            actions = torch.where(roll < epsilon, random_actions, q_values.argmax(dim=1)).to(
                torch.uint16
            )

        _ = self.env.step(actions, action_type=cfg.ROLL)
        return state, actions, mask

    def select_category_action(self, epsilon):
        # Reshape from [games_to_play, state_dim] -> [n_instances, games_per_instance, state_dim]
        state = self.env.get_encoded_state()[:, cfg.ROLL_SPECIFIC_INPUTS :]
        mask = self.env.get_category_action_masks()
        state_reshaped = state.view(self.n_instances, self.games_per_instance, -1)

        with torch.no_grad():
            q_values_reshaped = self.cat_forward(self.cat_params, self.cat_buffers, state_reshaped)
            q_values = q_values_reshaped.view(self.parallel_games, -1)

            q_values = q_values.masked_fill(~mask, -1e12)
            roll = torch.rand(q_values.size(0), device=device)
            random_actions = torch.rand_like(q_values).masked_fill(~mask, -1e12).argmax(dim=1)
            actions = torch.where(roll < epsilon, random_actions, q_values.argmax(dim=1)).to(
                torch.uint16
            )

        rewards = self.env.step(actions, action_type=cfg.CATEGORY)
        return state, actions, rewards, mask

    def play_with_roll_net(self):
        self.env = YahtzeeFast(self.parallel_games, self.device)
        T = cfg.ROLL_EXAMPLES_PER_GAME
        B = self.parallel_games
        S = self.env.ENCODED_STATE_SIZE
        P = self.roll_mem_pointer
        G = self.games_per_instance
        M = self.max_roll_states_per_instance
        N = self.roll_states_in_memory
        states = torch.zeros((T, B, S), device=self.device, dtype=torch.float)
        actions = torch.zeros((T, B), device=self.device, dtype=torch.float)
        masks = torch.zeros((T, B, S), device=self.device, dtype=torch.bool)
        round_rewards = torch.zeros((T // 2, B), device=self.device, dtype=torch.float)
        dones = torch.zeros((T, B), device=self.device, dtype=torch.bool)
        i = 0
        for r in range(cfg.ROUNDS_IN_GAME):
            for _ in range(2):
                state, action, mask = self.select_roll_action(self.epsilon)
                states[i] = state
                actions[i] = action
                masks[i] = mask
                dones[i] = torch.zeros(B, device=self.device)
                i += 1

            reward = self.select_category_action(0.0)
            round_rewards[r] = reward

        next_states = torch.zeros_like(states)
        next_masks = torch.zeros_like(masks)

        step_offset = self.n_step * 2

        # Shift the arrays backward by exactly `step_offset`
        next_states[:-step_offset] = states[step_offset:]
        next_masks[:-step_offset] = masks[step_offset:]

        # For the terminal transitions, backfill with the final observed state/mask
        next_states[-step_offset:] = state
        next_masks[-step_offset:] = mask

        padded_rewards = torch.cat(
            [round_rewards, torch.zeros((self.n_step, B), device=self.device, dtype=torch.float)],
            dim=0,
        )

        n_step_round_rewards = torch.zeros_like(round_rewards)

        for offset in range(self.n_step):
            discount = self.gamma**offset
            n_step_round_rewards += discount * padded_rewards[offset : offset + cfg.ROUNDS_IN_GAME]

        # Duplicate the round rewards to both sub-steps (action 1 and action 2)
        rewards = n_step_round_rewards.repeat_interleave(2, dim=0)

        X = self.n_instances
        B_sub = self.parallel_games // X

        def prepare_for_training(tensor):
            """
            Transforms: (T, B, S) -> (X, T * B_sub, S)
            """
            reshaped = tensor.view(T, X, B_sub, -1)
            permuted = reshaped.permute(1, 0, 2, 3)
            return permuted.reshape(X, T * B_sub, -1)

        states = prepare_for_training(states)
        actions = prepare_for_training(actions.unsqueeze(-1))
        rewards = prepare_for_training(rewards.unsqueeze(-1))
        dones = prepare_for_training(dones.unsqueeze(-1))
        next_states = prepare_for_training(next_states)
        next_masks = prepare_for_training(next_masks)

        if P + G <= M:  # New tensors fit
            self.roll_states[:, P : P + G] = states
            self.roll_actions[:, P : P + G] = actions
            self.roll_rewards[:, P : P + G] = rewards
            self.roll_dones[:, P : P + G] = dones
            self.roll_next_states[:, P : P + G] = next_states
            self.roll_next_masks[:, P : P + G] = next_masks

        else:  # New tensors have to be split
            # L is the amount of space remaining at the end of the buffer
            L = M - P

            # Fill the tail of the buffer with the first L elements
            self.roll_states[:, P:M] = states[:, :L]
            self.roll_actions[:, P:M] = actions[:, :L]
            self.roll_rewards[:, P:M] = rewards[:, :L]
            self.roll_dones[:, P:M] = dones[:, :L]
            self.roll_next_states[:, P:M] = next_states[:, :L]
            self.roll_next_masks[:, P:M] = next_masks[:, :L]

            # Wrap around to index 0 and write the remaining elements
            self.roll_states[:, 0 : G - L] = states[:, L:]
            self.roll_actions[:, 0 : G - L] = actions[:, L:]
            self.roll_rewards[:, 0 : G - L] = rewards[:, L:]
            self.roll_dones[:, 0 : G - L] = dones[:, L:]
            self.roll_next_states[:, 0 : G - L] = next_states[:, L:]
            self.roll_next_masks[:, 0 : G - L] = next_masks[:, L:]

        self.roll_mem_pointer = (P + G) % M
        N = min(N + G, M)
        print(f"Roll states in memory: {N}")

    def train_roll_net(self):
        pass


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
    parser.add_argument(
        "--n-instances",
        type=int,
        default=8,
        help="Number of parallel instances for ensemble training (must be a power of 2)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=8192,
        help="Number of parallel games to simulate for each training iteration "
        "(must be a power of 2)",
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=8192,
        help="Size of the training batch (must be a power of 2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Training learning rate",
    )
    parser.add_argument(
        "--learning-rate-gamma",
        type=float,
        default=1.0,
        help="The multiplier for learning rate for each iteration.",
    )
    parser.add_argument(
        "--training-loops", type=int, default=50, help="Number of training loops per iteration"
    )
    parser.add_argument(
        "--roll-cat-ratio",
        type=list[int],
        default=[4, 1],
        help="Ratio of roll and category networks training iterations. [2,1]"
        "means 2 training iterations for roll network for every 1 category"
        "training iterations. Default [4,1]",
    )
    parser.add_argument(
        "--epsilon-gamma",
        type=float,
        default=0.98,
        help="The multiplier for epsilon for each iteration. Default 0.98",
    )
    parser.add_argument(
        "--start-epsilon", type=float, default=1.0, help="The epsilon value at 1st iteration."
    )
    parser.add_argument(
        "--min-epsilon",
        type=float,
        default=0.001,
        help="The minimum value for epsilon during training.",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=2,
        help="N-step determines how many turns ahead the reward of a turn"
        "is calculated from. If N-step = 2 the immediater reward of a turn"
        "is the sum of 2 next category placements' rewards. Default: 2",
    )
    parser.add_argument(
        "--evaluation-frequency",
        type=int,
        default=10,
        help="How often the model is evaluated and the graphs are updated. Default: 10",
    )
    parser.add_argument(
        "--max-roll-states",
        type=int,
        default=200000,
        help="How many states, actions, rewards etc. are kept in memory at once for roll network"
        "Default: 200 000",
    )
    parser.add_argument(
        "--max-category-states",
        type=int,
        default=50000,
        help="How many states, actions, rewards etc. are kept in memory at once for category "
        "network. Default: 50 000",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Reward discount for each next step. Default: 0.99",
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
    trainer = TrainEnsemble(device, args)

    if args.mode == "train":
        trainer.train()

    if args.mode == "benchmark":
        trainer.benchmark()
