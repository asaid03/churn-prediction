import customtkinter as ctk
import tkinter as tk  
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AppView:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.create_panels()
        self.create_model_selection()
        self.create_menu_bar()

        # graphs
        self.graph_canvas = None

    def create_panels(self):
        """Create main panels for the GUI."""
        self.panel1 = ctk.CTkFrame(self.root)
        self.panel1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.panel1_title = ctk.CTkLabel(self.panel1, text="Model Performance Comparison", font=("Arial", 16))
        self.panel1_title.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.panel1.rowconfigure(4, weight=1)

        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

    def create_model_selection(self):
        """Create model selection checkboxes, visualisation selection and compare button."""
        self.checkbox_frame = ctk.CTkFrame(self.panel1)
        self.checkbox_frame.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_vars = {}

        vis_frame = ctk.CTkFrame(self.panel1)
        vis_frame.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        vis_label = ctk.CTkLabel(vis_frame, text="Select Visualisation:")
        vis_label.grid(row=0, column=0, padx=5, pady=5)
        self.visualisation_option = ctk.CTkOptionMenu(vis_frame, values=["Bar Graph", "Line Graph", "Correlation Matrix", "Exact Results"])
        self.visualisation_option.set("Bar Graph")
        self.visualisation_option.grid(row=0, column=1, padx=5, pady=5)

        self.compare_button = ctk.CTkButton(self.panel1, text="Compare Performance", command=self.on_compare_performance)
        self.compare_button.grid(row=3, column=0, padx=5, pady=10, sticky="w")

    def create_menu_bar(self):
        """Create the menu bar using tkinter's Menu widget."""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def show_about(self):
        """Display the about dialog."""
        messagebox.showinfo("About", "Churn Prediction GUI\nVersion 1.0")

    def update_model_list(self, models):
        """Update the checkbox list based on available models."""
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        self.model_vars = {}
        for idx, model in enumerate(models):
            var = ctk.BooleanVar()
            chk = ctk.CTkCheckBox(self.checkbox_frame, text=model, variable=var)
            chk.grid(row=idx, column=0, sticky="w", padx=5, pady=2)
            self.model_vars[model] = var

    def on_compare_performance(self):
        """Handle the 'Compare Performance' button click event."""
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one model.")
            return
        self.controller.compare_performance(selected_models)

    def display_graph(self, fig):
        """Display the given matplotlib figure in the GUI."""
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
        self.graph_canvas = FigureCanvasTkAgg(fig, master=self.panel1)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

    def display_performance_graph(self, data):
        """Display a bar graph of model performance."""
        fig, ax = plt.subplots()
        models = list(data.keys())
        metrics = list(next(iter(data.values())).keys())
        bar_width = 0.35
        for i, model in enumerate(models):
            metric_values = [data[model][metric] for metric in metrics]
            ax.bar([x + bar_width * i for x in range(len(metrics))], metric_values, bar_width, label=model)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks([x + bar_width for x in range(len(metrics))])
        ax.set_xticklabels(metrics)
        ax.legend()
        self.display_graph(fig)

    def display_line_graph(self, data):
        """Display a line graph of model performance by connecting metric values."""
        fig, ax = plt.subplots()
        for model, metric_dict in data.items():
            x = list(range(len(metric_dict)))
            y = list(metric_dict.values())
            ax.plot(x, y, marker='o', label=model)
        if data:
            metrics_keys = list(next(iter(data.values())).keys())
            ax.set_xticks(list(range(len(metrics_keys))))
            ax.set_xticklabels(metrics_keys)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Line Graph of Model Performance")
        ax.legend()
        self.display_graph(fig)

    def display_exact_results(self, data):
        """Display exact performance metrics in a formatted table."""
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
        result_frame = ctk.CTkFrame(self.panel1)
        result_frame.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        row_idx = 0
        for model, metrics in data.items():
            model_label = ctk.CTkLabel(result_frame, text=model, font=("Arial", 12, "bold"))
            model_label.grid(row=row_idx, column=0, padx=5, pady=5, sticky="w")
            col_idx = 1
            for metric, value in metrics.items():
                metric_label = ctk.CTkLabel(result_frame, text=f"{metric}: {value}", font=("Arial", 10))
                metric_label.grid(row=row_idx, column=col_idx, padx=5, pady=5, sticky="w")
                col_idx += 1
            row_idx += 1
        self.graph_canvas = result_frame

    def update(self, data):
        vis_type = self.visualisation_option.get()
        if vis_type == "Bar Graph":
            self.display_performance_graph(data)
        elif vis_type == "Line Graph":
            self.display_line_graph(data)
        elif vis_type == "Exact Results":
            self.display_exact_results(data)
        else:
            self.display_performance_graph(data)
