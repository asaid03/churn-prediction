import customtkinter as ctk
from gui.pages.model_performance import PerformancePage
from gui.pages.homepage import HomePage


class AppView:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller

        self.root.title("Churn Prediction GUI")
        self.root.geometry("1000x800")

        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        # Create tabs
        self.home_tab = self.tabview.add("Home")
        self.performance_tab = self.tabview.add("Performance")

    
        self.home_page = HomePage(self.home_tab)
        self.performance_page = PerformancePage(self.performance_tab, self.controller)

    def update_models(self, models):
        self.performance_page.update_models(models)

    def update(self, data):
        self.performance_page.show_results(data)
