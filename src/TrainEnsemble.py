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
        self.roll_training_batch_size = args.roll_training_batch_size
        self.cat_training_batch_size = args.cat_training_batch_size
        self.learning_rate = args.learning_rate
        self.learning_rate_gamma = args.learning_rate_gamma
        self.roll_training_loops = args.roll_training_loops
        self.cat_training_loops = args.cat_training_loops
        self.epsilon_gamma = args.epsilon_gamma
        self.start_epsilon = args.start_epsilon
        self.min_epsilon = args.min_epsilon
        self.n_step = args.n_step
        self.evaluation_frequency = args.evaluation_frequency
        self.max_roll_states = args.max_roll_states
        self.max_cat_states = args.max_category_states
        self.gamma = args.gamma
        self.start_tau = args.start_tau
        self.tau_gamma = args.tau_gamma
        self.min_tau = args.min_tau
        self.save_path = args.save_path
        self.entropy_multiplier = args.entropy_multiplier

        self.evaluating = False
        self.games_per_instance = self.parallel_games // self.n_instances
        self.max_roll_states_per_instance = self.max_roll_states // self.n_instances
        self.max_cat_states_per_instance = self.max_cat_states // self.n_instances
        self.roll_mem_pointer = 0
        self.cat_mem_pointer = 0
        self.roll_states_in_memory = 0
        self.cat_states_in_memory = 0
        self.best_seen_score = 0

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

    def save_models(self):
        """Save multiple networks to a single file."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        with torch.no_grad():
            for name, param in self.roll_policy_net.named_parameters():
                param.copy_(self.roll_params[name][self.best_instance_idx])
            for name, param in self.cat_policy_net.named_parameters():
                param.copy_(self.cat_params[name][self.best_instance_idx])

        checkpoint = {
            "roll_policy_net": self.roll_policy_net.state_dict(),
            "category_policy_net": self.cat_policy_net.state_dict(),
        }

        torch.save(checkpoint, self.save_path)
        print(f"Models saved to {self.save_path}")

    def setup_networks(self):
        self.roll_policy_net = RollPolicyNet().to(self.device)
        self.cat_policy_net = CategoryPolicyNet().to(self.device)
        self.target_roll_net = RollPolicyNet().to(self.device)
        self.target_cat_net = CategoryPolicyNet().to(self.device)
        if self.model_path:
            self.load_models(self.model_path)
            self.target_roll_net.load_state_dict(self.roll_policy_net.state_dict())
            self.target_cat_net.load_state_dict(self.cat_policy_net.state_dict())

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
        N = self.n_instances
        M = self.max_roll_states_per_instance
        roll_state_shape = (N, M, self.env.ENCODED_ROLL_STATE_SIZE)
        roll_mask_shape = (N, M, self.env.ROLL_ACTIONS)
        roll_action_shape = (N, M, 1)

        self.roll_states = torch.zeros(roll_state_shape, device=self.device, dtype=torch.float)
        self.roll_actions = torch.zeros(roll_action_shape, device=self.device, dtype=torch.int32)
        self.roll_rewards = torch.zeros(roll_action_shape, device=self.device, dtype=torch.int32)
        self.roll_next_states = torch.zeros(roll_state_shape, device=self.device, dtype=torch.float)
        self.roll_dones = torch.zeros(roll_action_shape, device=self.device, dtype=torch.bool)
        self.roll_next_masks = torch.zeros(roll_mask_shape, device=self.device, dtype=torch.bool)

        cat_state_shape = (N, M, self.env.ENCODED_CAT_STATE_SIZE)
        cat_mask_shape = (N, M, self.env.CAT_ACTIONS)
        cat_action_shape = (N, M, 1)

        self.cat_states = torch.zeros(cat_state_shape, device=self.device, dtype=torch.float)
        self.cat_actions = torch.zeros(cat_action_shape, device=self.device, dtype=torch.int32)
        self.cat_rewards = torch.zeros(cat_action_shape, device=self.device, dtype=torch.int32)
        self.cat_next_states = torch.zeros(cat_state_shape, device=self.device, dtype=torch.float)
        self.cat_dones = torch.zeros(cat_action_shape, device=self.device, dtype=torch.bool)
        self.cat_next_masks = torch.zeros(cat_mask_shape, device=self.device, dtype=torch.bool)

    def update_stats(self):
        """Update the wide figure with new batch statistics."""
        self.fig.suptitle(
            f"Games: {self.games_per_instance}, Avg Score: {self.best_avg_score:.2f}, "
            f"Best Seen Score: {self.best_seen_score:.2f}, "
            f"Bonuses: {self.stats['bonus_count']}, Yahtzees: {self.stats['yahtzee_count']} ",
            fontsize=16,
        )

        for i in range(16):
            # Pull distribution: index 0-14 for categories, roll_counts for index 15
            dist = self.stats["cat_distributions"][i] if i < 15 else self.stats["roll_counts"]

            # Vectorized update of bar heights
            for rect, h in zip(self.rects[i], dist):
                rect.set_height(h.item())

            # Rescale the Y-axis so bars don't go off-screen
            self.axes[i].relim()
            self.axes[i].autoscale_view(scalex=False, scaley=True)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def evaluate(self):
        print("Evaluating Network Performance")
        self.env = YahtzeeFast(self.parallel_games, self.device)
        T = cfg.ROLL_EXAMPLES_PER_GAME
        C = cfg.CATEGORY_EXAMPLES_PER_GAME
        B = self.parallel_games
        X = self.n_instances
        B_sub = B // X
        roll_actions = torch.zeros((T, B), device=self.device, dtype=torch.int32)
        cat_actions = torch.zeros((C, B), device=self.device, dtype=torch.int32)
        round_rewards = torch.zeros((C, B), device=self.device, dtype=torch.float)
        i = 0
        for r in range(cfg.ROUNDS_IN_GAME):
            for _ in range(2):
                _, action, _ = self.select_roll_action(0.0)
                roll_actions[i] = action
                i += 1
            _, action, _, reward = self.select_category_action(0.0)
            cat_actions[r] = action
            round_rewards[r] = reward

        def prepare_for_evaluation(tensor):
            """
            Transforms: (T, B, S) -> (X, T * B_sub, S)
            """
            T = tensor.shape[0]
            reshaped = tensor.view(T, X, B_sub, -1)
            permuted = reshaped.permute(1, 0, 2, 3)
            return permuted.reshape(X, T * B_sub, -1)

        roll_actions = prepare_for_evaluation(
            roll_actions.unsqueeze(-1),
        )
        cat_actions = prepare_for_evaluation(cat_actions.unsqueeze(-1))
        round_rewards = prepare_for_evaluation(round_rewards.unsqueeze(-1))

        average_scores = self.env.get_average_final_score_ensemble(X)
        best_instance_idx = average_scores.argmax().item()
        self.stats = self.env.analyze_batch_stats(
            roll_actions[best_instance_idx].flatten(),
            cat_actions[best_instance_idx].flatten(),
            round_rewards[best_instance_idx].flatten(),
        )

        self.best_avg_score = average_scores[best_instance_idx].item()
        if self.best_avg_score > self.best_seen_score:
            self.best_seen_score = self.best_avg_score
        self.best_instance_idx = best_instance_idx
        self.update_stats()

    def sync_to_best_instance(self):
        """
        Overwrites all instances in the batched ensemble with the
        weights and buffers of the target index. Also copies the states, actions, masks etc
        from the best instance to all the others.
        """
        with torch.no_grad():
            for param in self.roll_params.values():
                param.copy_(param[self.best_instance_idx].unsqueeze(0).expand_as(param))

            for buffer in self.roll_buffers.values():
                buffer.copy_(buffer[self.best_instance_idx].unsqueeze(0).expand_as(buffer))

            for param in self.cat_params.values():
                param.copy_(param[self.best_instance_idx].unsqueeze(0).expand_as(param))

            for buffer in self.cat_buffers.values():
                buffer.copy_(buffer[self.best_instance_idx].unsqueeze(0).expand_as(buffer))

            for opt in [self.roll_optimizer, self.cat_optimizer]:
                for param, state in opt.state.items():
                    if "exp_avg" in state:
                        state["exp_avg"].copy_(
                            state["exp_avg"][self.best_instance_idx]
                            .unsqueeze(0)
                            .expand_as(state["exp_avg"])
                        )
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].copy_(
                            state["exp_avg_sq"][self.best_instance_idx]
                            .unsqueeze(0)
                            .expand_as(state["exp_avg_sq"])
                        )

        instance_idx = torch.arange(self.n_instances, device=self.device).unsqueeze(1)
        self.roll_states[instance_idx] = self.roll_states[self.best_instance_idx].clone()
        self.roll_actions[instance_idx] = self.roll_actions[self.best_instance_idx].clone()
        self.roll_rewards[instance_idx] = self.roll_rewards[self.best_instance_idx].clone()
        self.roll_dones[instance_idx] = self.roll_dones[self.best_instance_idx].clone()
        self.roll_next_states[instance_idx] = self.roll_next_states[self.best_instance_idx].clone()
        self.roll_next_masks[instance_idx] = self.roll_next_masks[self.best_instance_idx].clone()
        self.cat_states[instance_idx] = self.cat_states[self.best_instance_idx].clone()
        self.cat_actions[instance_idx] = self.cat_actions[self.best_instance_idx].clone()
        self.cat_rewards[instance_idx] = self.cat_rewards[self.best_instance_idx].clone()
        self.cat_dones[instance_idx] = self.cat_dones[self.best_instance_idx].clone()
        self.cat_next_states[instance_idx] = self.cat_next_states[self.best_instance_idx].clone()
        self.cat_next_masks[instance_idx] = self.cat_next_masks[self.best_instance_idx].clone()

    def train(self):
        """Train the RollPolicyNet and CategoryPolicyNet using n parallel
        instances of the models so that on every x instances the best
        model is chosen and the rest of the models are updated with the
        state of the best model."""

        print(f"Training {self.n_instances} on device: {self.device}")
        for iteration in range(self.start_iter, self.num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.num_iterations} ===")
            self.epsilon = max(self.min_epsilon, self.epsilon_gamma**iteration)
            self.tau = max(self.min_tau, self.start_tau * self.tau_gamma**iteration)
            self.training_iteration()

            if (iteration + 1) % self.evaluation_frequency:
                continue

            self.evaluate()
            self.sync_to_best_instance()
            if self.best_avg_score == self.best_seen_score:
                self.save_models()

        plt.ioff()
        plt.show()

    def benchmark(self):
        """Run a benchmark of the current model without training."""
        print("Evaluating Network Performance")
        self.env = YahtzeeFast(self.parallel_games, self.device)
        T = cfg.ROLL_EXAMPLES_PER_GAME
        C = cfg.CATEGORY_EXAMPLES_PER_GAME
        B = self.parallel_games
        self.games_per_instance = B
        roll_actions = torch.zeros((T, B), device=self.device, dtype=torch.int32)
        cat_actions = torch.zeros((C, B), device=self.device, dtype=torch.int32)
        round_rewards = torch.zeros((C, B), device=self.device, dtype=torch.float)
        i = 0
        for r in range(cfg.ROUNDS_IN_GAME):
            for _ in range(2):
                state = self.env.get_encoded_state()
                mask = self.env.get_roll_action_masks()

                with torch.no_grad():
                    q_values = self.roll_policy_net(state).masked_fill(~mask, -1e12)
                    actions = q_values.argmax(dim=1).to(torch.int32)

                _ = self.env.step(actions, action_type=cfg.ROLL)
                roll_actions[i] = actions
                i += 1

            state = self.env.get_encoded_state()[:, cfg.ROLL_SPECIFIC_INPUTS :]
            mask = self.env.get_category_action_masks()

            with torch.no_grad():
                q_values = self.cat_policy_net(state).masked_fill(~mask, -1e12)
                actions = q_values.argmax(dim=1).to(torch.int32)

            reward = self.env.step(actions, action_type=cfg.CATEGORY)

            cat_actions[r] = actions
            round_rewards[r] = reward

        self.best_avg_score = self.env.get_average_final_score()
        self.stats = self.env.analyze_batch_stats(
            roll_actions.flatten(),
            cat_actions.flatten(),
            round_rewards.flatten(),
        )
        self.update_stats()
        print(f"Average score over {self.parallel_games} games: {self.best_avg_score:.2f}")
        plt.ioff()
        plt.show(block=True)

    def get_batched_loss_fn(self, base_model):
        """
        Creates a vectorized gradient function for N network instances.
        """

        def compute_single_loss(params, buffers, states, actions, targets):
            # Forward pass using functional_call (stateless)
            q_values = torch.func.functional_call(base_model, (params, buffers), states)
            q_value_preds = q_values.gather(1, actions)

            # Surprise-weighted logic
            with torch.no_grad():
                surprise = torch.clamp(targets - q_value_preds, min=0.0).detach()
                weight = 1.0 + (self.entropy_multiplier * surprise)

            raw_loss = torch.nn.functional.smooth_l1_loss(q_value_preds, targets, reduction="none")
            weighted_loss = (weight * raw_loss).mean()
            return weighted_loss

        # Vectorize across dimension 0 (the N_INSTANCES dimension) for all inputs
        # This will compute gradients for all N models simultaneously
        return torch.func.vmap(torch.func.grad(compute_single_loss), in_dims=(0, 0, 0, 0, 0))

    def training_iteration(self):
        """Do one training iteration."""
        print(
            f"Training with epsilon {self.epsilon:.3f}, "
            f"tau {self.tau:.3f}, "
            f"learning rate {self.roll_optimizer.param_groups[0]['lr']:.6f}"
        )
        self.play_games()
        self.train_roll_net()
        self.train_category_net()

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
            roll = torch.rand(q_values.size(0), device=self.device)
            random_actions = torch.rand_like(q_values).masked_fill(~mask, -1e12).argmax(dim=1)
            actions = torch.where(roll < epsilon, random_actions, q_values.argmax(dim=1)).to(
                torch.int32
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
            roll = torch.rand(q_values.size(0), device=self.device)
            random_actions = torch.rand_like(q_values).masked_fill(~mask, -1e12).argmax(dim=1)
            actions = torch.where(roll < epsilon, random_actions, q_values.argmax(dim=1)).to(
                torch.int32
            )

        reward = self.env.step(actions, action_type=cfg.CATEGORY)
        return state, actions, mask, reward

    def play_games(self):
        self.env = YahtzeeFast(self.parallel_games, self.device)
        RE = cfg.ROLL_EXAMPLES_PER_GAME
        CE = cfg.CATEGORY_EXAMPLES_PER_GAME
        P = self.parallel_games
        RS = self.env.ENCODED_ROLL_STATE_SIZE
        CS = self.env.ENCODED_CAT_STATE_SIZE
        RA = self.env.ROLL_ACTIONS
        CA = self.env.CAT_ACTIONS
        G = self.games_per_instance

        roll_states = torch.zeros((RE, P, RS), device=self.device, dtype=torch.float)
        roll_actions = torch.zeros((RE, P), device=self.device, dtype=torch.float)
        roll_masks = torch.zeros((RE, P, RA), device=self.device, dtype=torch.bool)
        roll_dones = torch.zeros((RE, P), device=self.device, dtype=torch.bool)
        cat_states = torch.zeros((CE, P, CS), device=self.device, dtype=torch.float)
        cat_actions = torch.zeros((CE, P), device=self.device, dtype=torch.float)
        cat_masks = torch.zeros((CE, P, CA), device=self.device, dtype=torch.bool)
        cat_dones = torch.zeros((CE, P), device=self.device, dtype=torch.bool)
        round_rewards = torch.zeros((CE, P), device=self.device, dtype=torch.float)

        i = 0
        for r in range(cfg.ROUNDS_IN_GAME):
            for _ in range(2):
                roll_state, roll_action, roll_mask = self.select_roll_action(self.epsilon)
                roll_states[i] = roll_state
                roll_actions[i] = roll_action
                roll_masks[i] = roll_mask
                i += 1

            cat_state, cat_action, cat_mask, reward = self.select_category_action(self.epsilon)
            cat_states[r] = cat_state
            cat_actions[r] = cat_action
            cat_masks[r] = cat_mask
            round_rewards[r] = reward

        roll_next_states = torch.zeros_like(roll_states)
        roll_next_masks = torch.zeros_like(roll_masks)
        cat_next_states = torch.zeros_like(cat_states)
        cat_next_masks = torch.zeros_like(cat_masks)

        roll_offset = self.n_step * 2
        cat_offset = self.n_step

        roll_dones[-roll_offset:] = torch.ones(P, device=self.device)
        cat_dones[-cat_offset:] = torch.ones(P, device=self.device)

        # Shift the arrays backward by exactly `step_offset`
        roll_next_states[:-roll_offset] = roll_states[roll_offset:]
        roll_next_masks[:-roll_offset] = roll_masks[roll_offset:]
        cat_next_states[:-cat_offset] = cat_states[cat_offset:]
        cat_next_masks[:-cat_offset] = cat_masks[cat_offset:]

        # For the terminal transitions, backfill with the final observed state/mask
        roll_next_states[-roll_offset:] = roll_state
        roll_next_masks[-roll_offset:] = roll_mask
        cat_next_states[-cat_offset:] = cat_state
        cat_next_masks[-cat_offset:] = cat_mask

        padded_rewards = torch.cat(
            [round_rewards, torch.zeros((self.n_step, P), device=self.device, dtype=torch.float)],
            dim=0,
        )

        n_step_round_rewards = torch.zeros_like(round_rewards)

        for offset in range(self.n_step):
            discount = self.gamma**offset
            n_step_round_rewards += discount * padded_rewards[offset : offset + cfg.ROUNDS_IN_GAME]

        # Duplicate the round rewards to both sub-steps (action 1 and action 2)
        roll_rewards = n_step_round_rewards.repeat_interleave(2, dim=0)
        cat_rewards = n_step_round_rewards

        X = self.n_instances
        B_sub = self.parallel_games // X

        def prepare_for_training(tensor):
            """
            Transforms: (T, B, S) -> (X, T * B_sub, S)
            """
            T = tensor.shape[0]
            reshaped = tensor.view(T, X, B_sub, -1)
            permuted = reshaped.permute(1, 0, 2, 3)
            return permuted.reshape(X, T * B_sub, -1)

        roll_states = prepare_for_training(roll_states)
        roll_actions = prepare_for_training(roll_actions.unsqueeze(-1))
        roll_rewards = prepare_for_training(roll_rewards.unsqueeze(-1))
        roll_dones = prepare_for_training(roll_dones.unsqueeze(-1))
        roll_next_states = prepare_for_training(roll_next_states)
        roll_next_masks = prepare_for_training(roll_next_masks)
        cat_states = prepare_for_training(cat_states)
        cat_actions = prepare_for_training(cat_actions.unsqueeze(-1))
        cat_rewards = prepare_for_training(cat_rewards.unsqueeze(-1))
        cat_dones = prepare_for_training(cat_dones.unsqueeze(-1))
        cat_next_states = prepare_for_training(cat_next_states)
        cat_next_masks = prepare_for_training(cat_next_masks)

        RM = self.max_roll_states_per_instance
        RP = self.roll_mem_pointer
        if RP + G * RE <= RM:
            self.roll_states[:, RP : RP + G * RE] = roll_states
            self.roll_actions[:, RP : RP + G * RE] = roll_actions
            self.roll_rewards[:, RP : RP + G * RE] = roll_rewards
            self.roll_dones[:, RP : RP + G * RE] = roll_dones
            self.roll_next_states[:, RP : RP + G * RE] = roll_next_states
            self.roll_next_masks[:, RP : RP + G * RE] = roll_next_masks

        else:  # New tensors have to be split
            # L is the amount of space remaining at the end of the buffer
            L = RM - RP

            # Fill the tail of the buffer with the first L elements
            self.roll_states[:, RP:RM] = roll_states[:, :L]
            self.roll_actions[:, RP:RM] = roll_actions[:, :L]
            self.roll_rewards[:, RP:RM] = roll_rewards[:, :L]
            self.roll_dones[:, RP:RM] = roll_dones[:, :L]
            self.roll_next_states[:, RP:RM] = roll_next_states[:, :L]
            self.roll_next_masks[:, RP:RM] = roll_next_masks[:, :L]

            # Wrap around to index 0 and write the remaining elements
            self.roll_states[:, 0 : G * RE - L] = roll_states[:, L:]
            self.roll_actions[:, 0 : G * RE - L] = roll_actions[:, L:]
            self.roll_rewards[:, 0 : G * RE - L] = roll_rewards[:, L:]
            self.roll_dones[:, 0 : G * RE - L] = roll_dones[:, L:]
            self.roll_next_states[:, 0 : G * RE - L] = roll_next_states[:, L:]
            self.roll_next_masks[:, 0 : G * RE - L] = roll_next_masks[:, L:]

        CM = self.max_cat_states_per_instance
        CP = self.cat_mem_pointer
        if CP + G * CE <= CM:
            self.cat_states[:, CP : CP + G * CE] = cat_states
            self.cat_actions[:, CP : CP + G * CE] = cat_actions
            self.cat_rewards[:, CP : CP + G * CE] = cat_rewards
            self.cat_dones[:, CP : CP + G * CE] = cat_dones
            self.cat_next_states[:, CP : CP + G * CE] = cat_next_states
            self.cat_next_masks[:, CP : CP + G * CE] = cat_next_masks

        else:  # New tensors have to be split
            # L is the amount of space remaining at the end of the buffer
            L = CM - CP

            # Fill the tail of the buffer with the first L elements
            self.cat_states[:, CP:CM] = cat_states[:, :L]
            self.cat_actions[:, CP:CM] = cat_actions[:, :L]
            self.cat_rewards[:, CP:CM] = cat_rewards[:, :L]
            self.cat_dones[:, CP:CM] = cat_dones[:, :L]
            self.cat_next_states[:, CP:CM] = cat_next_states[:, :L]
            self.cat_next_masks[:, CP:CM] = cat_next_masks[:, :L]

            # Wrap around to index 0 and write the remaining elements
            self.cat_states[:, 0 : G * CE - L] = cat_states[:, L:]
            self.cat_actions[:, 0 : G * CE - L] = cat_actions[:, L:]
            self.cat_rewards[:, 0 : G * CE - L] = cat_rewards[:, L:]
            self.cat_dones[:, 0 : G * CE - L] = cat_dones[:, L:]
            self.cat_next_states[:, 0 : G * CE - L] = cat_next_states[:, L:]
            self.cat_next_masks[:, 0 : G * CE - L] = cat_next_masks[:, L:]

        RN = self.roll_states_in_memory
        CN = self.cat_states_in_memory
        self.roll_mem_pointer = (RP + G * RE) % RM
        self.roll_states_in_memory = min(RN + G * RE * self.n_instances, self.max_roll_states)
        self.cat_mem_pointer = (CP + G * CE) % CM
        self.cat_states_in_memory = min(CN + G * CE * self.n_instances, self.max_cat_states)
        avg_score = self.env.get_average_final_score()
        print(f"Played {self.parallel_games} parallel games. Average score: {avg_score:.2f}")
        print(f"Roll states in memory: {self.roll_states_in_memory}")
        print(f"Category states in memory: {self.cat_states_in_memory}")

    def calculate_roll_targets(self):
        with torch.no_grad():
            next_q_values = self.roll_target_forward(
                self.roll_target_params, self.roll_target_buffers, self.next_states_batch
            )
            next_q_values = next_q_values.masked_fill(~self.next_masks_batch.bool(), -1e12)
            max_next_q = next_q_values.max(dim=-1, keepdim=True)[0]
            self.targets = self.rewards_batch + max_next_q * (~self.dones_batch) * (
                self.gamma**self.n_step
            )

    def calculate_category_targets(self):
        with torch.no_grad():
            next_q_values = self.cat_target_forward(
                self.cat_target_params, self.cat_target_buffers, self.next_states_batch
            )
            next_q_values = next_q_values.masked_fill(~self.next_masks_batch.bool(), -1e12)
            max_next_q = next_q_values.max(dim=-1, keepdim=True)[0]
            self.targets = self.rewards_batch + max_next_q * (~self.dones_batch) * (
                self.gamma**self.n_step
            )

    def train_roll_batch(self):
        self.roll_optimizer.zero_grad()
        batched_grads = self.compute_grads(
            self.roll_params, self.roll_buffers, self.states_batch, self.actions_batch, self.targets
        )
        for key, param in self.roll_params.items():
            param.grad = batched_grads[key]

        self.roll_optimizer.step()

    def train_category_batch(self):
        self.cat_optimizer.zero_grad()
        batched_grads = self.compute_grads(
            self.cat_params, self.cat_buffers, self.states_batch, self.actions_batch, self.targets
        )
        for key, param in self.cat_params.items():
            param.grad = batched_grads[key]

        self.cat_optimizer.step()

    def soft_update_roll_target(self):
        with torch.no_grad():
            for key in self.roll_params.keys():
                self.roll_target_params[key].mul_(1.0 - self.tau).add_(
                    self.roll_params[key], alpha=self.tau
                )

            for key in self.roll_buffers.keys():
                if self.roll_target_buffers[key].is_floating_point():
                    self.roll_target_buffers[key].mul_(1.0 - self.tau).add_(
                        self.roll_buffers[key], alpha=self.tau
                    )
                else:
                    self.roll_target_buffers[key].copy_(self.roll_buffers[key])

    def soft_update_category_target(self):
        with torch.no_grad():
            for key in self.cat_params.keys():
                self.cat_target_params[key].mul_(1.0 - self.tau).add_(
                    self.cat_params[key], alpha=self.tau
                )

            for key in self.cat_buffers.keys():
                if self.cat_target_buffers[key].is_floating_point():
                    self.cat_target_buffers[key].mul_(1.0 - self.tau).add_(
                        self.cat_buffers[key], alpha=self.tau
                    )
                else:
                    self.cat_target_buffers[key].copy_(self.cat_buffers[key])

    def train_roll_net(self):
        self.compute_grads = self.get_batched_loss_fn(self.roll_policy_net)
        instance_idx = torch.arange(self.n_instances, device=self.device).unsqueeze(1)
        for _ in range(self.roll_training_loops):
            sample_indices = torch.randint(
                low=0,
                high=self.roll_states_in_memory // self.n_instances,
                size=(self.n_instances, self.roll_training_batch_size // self.n_instances),
                device=self.device,
            )
            self.states_batch = self.roll_states[instance_idx, sample_indices]
            self.actions_batch = self.roll_actions[instance_idx, sample_indices]
            self.rewards_batch = self.roll_rewards[instance_idx, sample_indices]
            self.dones_batch = self.roll_dones[instance_idx, sample_indices]
            self.next_states_batch = self.roll_next_states[instance_idx, sample_indices]
            self.next_masks_batch = self.roll_next_masks[instance_idx, sample_indices]
            self.calculate_roll_targets()
            self.train_roll_batch()
        self.soft_update_roll_target()
        self.roll_scheduler.step()

    def train_category_net(self):
        self.compute_grads = self.get_batched_loss_fn(self.cat_policy_net)
        instance_idx = torch.arange(self.n_instances, device=self.device).unsqueeze(1)
        for _ in range(self.cat_training_loops):
            sample_indices = torch.randint(
                low=0,
                high=self.cat_states_in_memory // self.n_instances,
                size=(self.n_instances, self.cat_training_batch_size // self.n_instances),
                device=self.device,
            )
            self.states_batch = self.cat_states[instance_idx, sample_indices]
            self.actions_batch = self.cat_actions[instance_idx, sample_indices]
            self.rewards_batch = self.cat_rewards[instance_idx, sample_indices]
            self.dones_batch = self.cat_dones[instance_idx, sample_indices]
            self.next_states_batch = self.cat_next_states[instance_idx, sample_indices]
            self.next_masks_batch = self.cat_next_masks[instance_idx, sample_indices]
            self.calculate_category_targets()
            self.train_category_batch()
        self.soft_update_category_target()
        self.cat_scheduler.step()


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
        "--roll-training-batch-size",
        type=int,
        default=8192,
        help="Size of the training batch for roll network (must be a power of 2)",
    )
    parser.add_argument(
        "--cat-training-batch-size",
        type=int,
        default=8192,
        help="Size of the training batch for category network (must be a power of 2)",
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
        "--roll-training-loops",
        type=int,
        default=40,
        help="Number of training loops per iteration for roll selection network.",
    )
    parser.add_argument(
        "--cat-training-loops",
        type=int,
        default=20,
        help="Number of training loops per iteration for category selection network.",
    )
    parser.add_argument(
        "--epsilon-gamma",
        type=float,
        default=0.99,
        help="The multiplier for epsilon for each iteration. Default 0.99",
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
        default=3,
        help="N-step determines how many turns ahead the reward of a turn"
        "is calculated from. If N-step = 2 the immediater reward of a turn"
        "is the sum of 2 next category placements' rewards. Default: 3",
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
        default=2000000,
        help="How many states, actions, rewards etc. are kept in memory at once for roll network"
        "Default: 2 000 000",
    )
    parser.add_argument(
        "--max-category-states",
        type=int,
        default=1000000,
        help="How many states, actions, rewards etc. are kept in memory at once for category "
        "network. Default: 1 000 000",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Reward discount for each next step. Default: 0.99",
    )
    parser.add_argument(
        "--start-tau",
        type=float,
        default=0.2,
        help="Determines how big of a proportion of the trained networks properties. "
        "are transformed to the target network per iteration. Default: 0.2",
    )
    parser.add_argument(
        "--tau-gamma",
        type=float,
        default=0.998,
        help="The multiplier for tau for each iteration. Default: 0.998",
    )
    parser.add_argument(
        "--min-tau",
        type=float,
        default=0.01,
        help="The minimum value for tau during training. Default: 0.01",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/best_model_ensemble.pth",
        help="Where to save the best model after each evaluation. Default: "
        "models/best_model_ensemble.pth",
    )
    parser.add_argument(
        "--entropy-multiplier",
        type=float,
        default=0.01,
        help="Multiplier for the surprise-weighted loss. Default: 0.01",
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
