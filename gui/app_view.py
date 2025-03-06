import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class AppView:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.root.title("Model Analytics")
        self.root.geometry("1000x600")
    
        # Configure main layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        self._create_sidebar()
        self._create_viz_area()

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self.root, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Model selection
        ctk.CTkLabel(self.sidebar, text="Select Models:", anchor="w").pack(pady=(20,5), padx=20)
        self.model_vars = {}
        self.models_frame = ctk.CTkScrollableFrame(self.sidebar)
        self.models_frame.pack(fill="both", expand=True, padx=20)
        
        # Visualisation type
        self.viz_type = ctk.CTkOptionMenu(self.sidebar, values=["Bar Chart", "Line Chart", "Exact Values"])
        self.viz_type.set("Bar Chart")
        self.viz_type.pack(pady=10, padx=20, fill="x")
        
        # Compare button
        ctk.CTkButton(self.sidebar, text="Compare", command=self._handle_compare).pack(pady=20, padx=20, fill="x")

    def _create_viz_area(self):
        """Create main visualisation canvas"""
        self.viz_frame = ctk.CTkFrame(self.root)
        self.viz_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.graph_canvas = None
        

    def update_models(self, models):
        """Update model checkboxes"""
        for widget in self.models_frame.winfo_children():
            widget.destroy()
        self.model_vars = {model: ctk.BooleanVar() for model in models}
        for model, var in self.model_vars.items():
            ctk.CTkCheckBox(self.models_frame, text=model, variable=var).pack(anchor="w")

    def _handle_compare(self):
        """Handles comparison action by validating selection and triggering performance comparison."""
        selected_models = []
        
        for model_name, checkbox_var in self.model_vars.items():
            if checkbox_var.get():
                selected_models.append(model_name)
                
        if not selected_models:
            ctk.CTkMessagebox.show_error(title="Selection Needed",message="Please select at least one model before comparing.")
            return
        self.controller.compare_performance(selected_models)


    def show_results(self, data):
        """Display results with consistent layout reset."""
        
        # Clear out all existing widgets
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        self.graph_canvas = None
        self.viz_frame.grid_rowconfigure(0, weight=1)
        self.viz_frame.grid_columnconfigure(0, weight=1)

        viz_type = self.viz_type.get()
        viz_methods = {
            "Bar Chart": self._show_bar_chart,
            "Line Chart": self._show_line_chart,
            "Exact Values": self._show_table
        }

        viz_method = viz_methods.get(viz_type, self._show_bar_chart)
        viz_method(data)
        
        
    def _show_line_chart(self, data):
        """line chart visualisation"""
        fig, ax = plt.subplots(figsize=(8, 4))
        for model, metrics in data.items():
            ax.plot(list(metrics.values()), marker='o', label=model)
            
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics.keys(), rotation=25)
        ax.legend()
        self._display_figure(fig)
        

    def _show_bar_chart(self, data):
        """Display model performance as a grouped bar chart using DataFrame."""
        if self.graph_canvas:
            self.graph_canvas.destroy()

        #plotting
        df = pd.DataFrame(data).T
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
        df.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.7)

        # AXIS/LABELS
        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Values", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, pad=20)
        ax.legend(title="Metrics", bbox_to_anchor=(1, 1), loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        fig.tight_layout()
        self._display_figure(fig)



    def _show_table(self, data):
        """Display model performance in a scrollable table."""
        if self.graph_canvas:
            self.graph_canvas.destroy()  # Clear previous table
        self.graph_canvas = ctk.CTkScrollableFrame(self.viz_frame, label_text="Performance Metrics")
        self.graph_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Create table headers
        headers = ["Model"] + list(data[next(iter(data))].keys())
        for col in range(len(headers)):
            label = ctk.CTkLabel(self.graph_canvas, text=headers[col], font=("Arial", 12, "bold"))
            label.grid(row=0, column=col, padx=10, pady=5)

        # Add table rows
        row = 1
        for model, metrics in data.items():
            ctk.CTkLabel(self.graph_canvas, text=model, font=("Arial", 11)).grid(row=row, column=0, padx=10, pady=5)
            col = 1
            for value in metrics.values():
                ctk.CTkLabel(self.graph_canvas, text=f"{value:.4f}", font=("Arial", 11)).grid(row=row, column=col, padx=10, pady=5)
                col += 1
            row += 1


    def _display_figure(self, fig):
        """Helper to display matplotlib figures"""
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
        self.graph_canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def update(self, data):
        """Observer update method called by the model"""
        self.show_results(data)
