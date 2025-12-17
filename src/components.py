import tkinter as tk


def enable_neural_network_button(app):
    def enable_nn():
        # Attempt to load model
        if app.game.load_model():
            app.nn_button.config(state="disabled", text="AI Active")
            app.ai_label.config(text="AI initialized. Roll to start.")
            # Trigger initial suggestion if game is in progress
            update_ai_suggestion(app)
        else:
            app.ai_label.config(text="Model not found in models/")

    app.nn_button = tk.Button(
        app.window, text="Enable Neural Network", command=enable_nn
    )
    app.nn_button.pack(pady=5)

    # Label to display AI suggestions
    app.ai_label = tk.Label(
        app.window,
        text="AI Suggestions: Off",
        font=("Helvetica", 12, "bold"),
        fg="blue",
    )
    app.ai_label.pack(pady=5)


def update_ai_suggestion(app):
    """Queries the AI and updates the suggestion label."""
    if app.game.neural_network:
        if app.game.rolls_left < 3:
            suggestion = app.game.get_ai_prediction()
            app.ai_label.config(text=f"AI Suggests: {suggestion}")
        else:
            app.ai_label.config(text="AI Suggestions: Roll to start")


def roll_button(app):
    def roll_dice():
        app.game.roll_dice()
        dice = app.game.dice
        for i, value in enumerate(dice):
            app.dice_buttons[i].config(text=str(value))
            if app.game.rolls_left == 2:  # First roll of turn
                app.dice_buttons[i].config(state="normal", bg="white")
            elif app.game.rolls_left == 0:
                app.dice_buttons[i].config(state="disabled", bg="white")

        if app.game.rolls_left == 2:
            for btn in app.dice_buttons:
                btn.config(borderwidth=2, relief="flat")

        _update_scores(app)
        update_ai_suggestion(app)

    app.roll_button = tk.Button(app.window, text="Roll Dice", command=roll_dice)
    app.roll_button.pack(pady=10)


def dice_buttons(app):
    def dice_action(index):
        button = app.dice_buttons[index]
        if button.cget("relief") == "flat":
            button.config(borderwidth=2, relief="solid")
            app.game.lock_dice(index)
        else:
            button.config(borderwidth=2, relief="flat")
            app.game.unlock_dice(index)

    app.dice_buttons = [
        tk.Button(
            app.window,
            text="0",
            font=("Helvetica", 24),
            command=lambda i=i: dice_action(i),
            state="disabled",
            relief="flat",
            borderwidth=2,
        )
        for i in range(5)
    ]
    for button in app.dice_buttons:
        button.pack(side=tk.LEFT, padx=5)


def category_frame(app):
    def select_category(category):
        app.game.select_category(category)
        _update_scores(app)
        _reset_dice(app)
        update_ai_suggestion(app)

    frame = tk.Frame(app.window)
    frame.pack()

    app.category_buttons = []
    app.category_score_labels = []

    # Use keys from game instance to ensure order matching
    for category in app.game.category_names:
        category_frame = tk.Frame(frame)
        category_frame.pack(pady=1)
        if category == "bonus":
            app.upper_categories_label = tk.Label(
                category_frame,
                text="Upper Section: 0",
                font=("Helvetica", 16),
            )
            app.bonus_label = tk.Label(
                category_frame,
                text="Bonus: 0",
                font=("Helvetica", 16),
            )
            app.upper_categories_label.pack(pady=1)
            app.bonus_label.pack(pady=1)
            continue

        button = tk.Button(
            category_frame,
            text=f"{category.replace('_', ' ').title()}: 0",
            command=lambda c=category: select_category(c),
            state="disabled",
            width=20,
            font=("Helvetica", 14),
        )
        button.pack(side=tk.LEFT)
        app.category_buttons.append(button)

        score_label = tk.Label(
            category_frame, text="0", font=("Helvetica", 14), width=5
        )
        score_label.pack(side=tk.RIGHT)
        app.category_score_labels.append(score_label)


def total_score_label(app):
    app.score_label = tk.Label(
        app.window, text="Total Score: 0", font=("Helvetica", 16)
    )
    app.score_label.pack(pady=10)


def _update_scores(app):
    """Update the categories buttons"""
    # Map button list index to category names, skipping bonus logic which is handled separately in layout
    # app.category_buttons contains 15 buttons (0-14).
    # game.category_names has 16 items. 'bonus' is index 15.

    cat_idx = 0
    for name in app.game.category_names:
        value = app.game.categories[name]

        if name == "bonus":
            app.upper_categories_label.config(
                text=f"Upper Section: {app.game.get_upper_section_score()}"
            )
            app.bonus_label.config(text=f"Bonus: {value if value is not None else 0}")
            continue

        button = app.category_buttons[cat_idx]
        score_preview = app.game.calculate_score(name)

        # If value is set (played), show that. Else show preview of current dice
        display_score = value if value is not None else score_preview

        button.config(text=f"{name.replace('_', ' ').title()}: {display_score}")
        app.category_score_labels[cat_idx].config(
            text=str(value if value is not None else 0)
        )

        # Enable button only if not played and game not over
        if value is None:
            button.config(state="normal")
        else:
            button.config(state="disabled")

        cat_idx += 1

    # Update overall score
    app.score_label.config(text=f"Total Score: {app.game.get_score()}")


def _reset_dice(app):
    """Reset the dice buttons"""
    for button in app.dice_buttons:
        button.config(state="disabled", bg="white", borderwidth=2, relief="flat")
        button.config(text="0")
    for button in app.category_buttons:
        button.config(state="disabled")
    app.game.reset_dice()
