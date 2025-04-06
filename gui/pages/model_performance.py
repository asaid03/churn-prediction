import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

class PerformancePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.pack(expand=True, fill="both")
        self.controller = controller

        self._create_sidebar()
        self._create_visual_panel()

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300)
        self.sidebar.pack(side="left", fill="y")
        
        
        ctk.CTkLabel(self.sidebar, text="Performance Type:", anchor="w").pack(pady=(10, 0), padx=20)
        
        # OPt between test set or cross-validation
        self.performance_type = ctk.CTkOptionMenu(self.sidebar,values=["Test Set", "Cross-Validation"])
        self.performance_type.set("Test Set")
        self.performance_type.pack(pady=5, padx=20, fill="x")

        
        # filter optionss
        ctk.CTkLabel(self.sidebar, text="Filter:", anchor="w").pack(pady=(20, 5), padx=20)
        self.resample_filter = ctk.CTkOptionMenu(
            self.sidebar,
            values=["All", "Original", "SMOTE", "Tomek"],
            command=self._handle_resample_filter
        )
        self.resample_filter.set("All")
        self.resample_filter.pack(pady=5, padx=20, fill="x")

        
        ctk.CTkLabel(self.sidebar, text="Select Models:", anchor="w").pack(pady=(20, 5), padx=20)
        self.model_vars = {}
        self.models_frame = ctk.CTkScrollableFrame(self.sidebar)
        self.models_frame.pack(fill="both", expand=True, padx=20)

        ctk.CTkLabel(self.sidebar, text="View Type", anchor="w").pack(pady=(10, 0), padx=20)
        self.viz_type = ctk.CTkOptionMenu(self.sidebar, values=["Bar Chart", "Radar Chart", "Exact Values"])
        self.viz_type.set("Bar Chart")
        self.viz_type.pack(pady=5, padx=20, fill="x")

        self.compare_button = ctk.CTkButton(self.sidebar, text="Compare", command=self._compare_models)
        self.compare_button.pack(pady=20, padx=20, fill="x")

    def _create_visual_panel(self):
        self.main_area = ctk.CTkFrame(self)
        self.main_area.pack(side="left", expand=True, fill="both", padx=20, pady=20)
        self.plot_area = None

    def update_models(self, models):
        self.all_models = models  # all models
        self._filter_and_display_models()  #filtering
        
    def _handle_resample_filter(self, filter):
        self._filter_and_display_models()
   

    def _filter_and_display_models(self):
        for widget in self.models_frame.winfo_children():
            widget.destroy()

        self.model_vars = {}
        resample = self.resample_filter.get()

        for model in self.all_models:
            if resample == "All":
                pass_filter = True
            elif resample == "Original":
                pass_filter = "[" not in model
            else:
                pass_filter = f"[{resample}]" in model

            if pass_filter:
                var = ctk.BooleanVar()
                checkbox = ctk.CTkCheckBox(self.models_frame, text=model, variable=var)
                checkbox.pack(anchor="w")
                self.model_vars[model] = var


    def _compare_models(self):
        selected = [name for name, var in self.model_vars.items() if var.get()]
        if not selected:
            tk.messagebox.showerror("No Models Selected", "Please select at least one model.")
            return
        perf_type = self.performance_type.get()
        self.controller.compare_performance(selected, perf_type)

    def show_results(self, data):
        for widget in self.main_area.winfo_children():
            widget.destroy()

        self.plot_area = None
        viz_choice = self.viz_type.get()
        if viz_choice == "Bar Chart":
            self._show_bar_chart(data)
        elif viz_choice == "Radar Chart":
            self._show_radar_chart(data)
        else:
            self._show_table(data)

    def _show_bar_chart(self, data):
        df = pd.DataFrame(data).T
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.7)

        ax.set_title("Model Performance", fontsize=14)
        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Scores", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title="Metrics", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        self._display_figure(fig)
        

        
    def _show_radar_chart(self, data):
        df = pd.DataFrame(data).T

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="polar")

        labels = df.columns.tolist()
        theta = np.arange(len(labels) + 1) / float(len(labels)) * 2 * np.pi

        for model in df.index:
            values = df.loc[model].values
            values = np.append(values, values[0])

            ax.plot(theta, values, marker="o", label=model)
            ax.fill(theta, values, alpha=0.1)

        ax.set_xticks(theta[:-1])
        ax.set_xticklabels(labels, color="grey", size=12)
        ax.tick_params(pad=10)
        ax.legend(loc="best")
        self._display_figure(fig)


    def _show_table(self, data):
        self.plot_area = ctk.CTkScrollableFrame(self.main_area, label_text="Model Metrics")
        self.plot_area.pack(fill="both", expand=True, padx=10, pady=10)

        headers = ["Model"] + list(next(iter(data.values())).keys())
        for col, header in enumerate(headers):
            ctk.CTkLabel(self.plot_area, text=header, font=("Arial", 12, "bold")).grid(row=0, column=col, padx=10, pady=5)

        for row, (model, metrics) in enumerate(data.items(), start=1):
            ctk.CTkLabel(self.plot_area, text=model).grid(row=row, column=0, padx=10, pady=5)
            for col, val in enumerate(metrics.values(), start=1):
                ctk.CTkLabel(self.plot_area, text=f"{val:.4f}").grid(row=row, column=col, padx=10, pady=5)

    def _display_figure(self, fig):
        if self.plot_area:
            try:
                self.plot_area.get_tk_widget().destroy()
            except AttributeError:
                self.plot_area.destroy()

        self.plot_area = FigureCanvasTkAgg(fig, master=self.main_area)
        self.plot_area.draw()
        self.plot_area.get_tk_widget().pack(fill="both", expand=True)
        self.models_frame.update_idletasks()

    def update(self, data):
        self.show_results(data)
        
