import tkinter as tk

from components import category_frame, dice_buttons, roll_button, total_score_label
from Yahtzee import Yahtzee


class YahtzeeApp:
    def __init__(self, root):
        self.window = root
        self.window.title("Yahtzee Game")
        self.game = Yahtzee()

        self.init()

    def init(self):
        roll_button(self)
        dice_buttons(self)
        category_frame(self)
        total_score_label(self)


if __name__ == "__main__":
    root = tk.Tk()
    app = YahtzeeApp(root)

    # Set the window to appear on a specific position
    root.geometry("+1000+100")

    root.mainloop()
