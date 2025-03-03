import customtkinter as ctk
from gui.app_model import AppModel
from gui.app_view import AppView
from gui.app_controller import AppController
import sys

def main():
    # Use CTk() from customtkinter for a modern look
    root = ctk.CTk()
    root.geometry("900x700")
    root.title("Churn Prediction GUI")

    def on_closing():
        print("Closing application...")
        root.quit()
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Initialise Model, View, Controller
    model = AppModel()
    view = AppView(root, None)
    controller = AppController(model, view)
    view.controller = controller

    # Populate the model list in the view
    view.update_model_list(list(model.models.keys()))

    root.mainloop()

if __name__ == '__main__':
    main()
