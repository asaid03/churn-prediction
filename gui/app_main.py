import customtkinter as ctk
from gui.app_model import AppModel
from gui.app_controller import AppController
from gui.pages.model_performance import PerformancePage
import sys

def main():
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
    view = PerformancePage(root, None)
    controller = AppController(model, view)
    view.controller = controller

    view.update_models(model.models.keys())

    root.mainloop()

if __name__ == '__main__':
    main()
