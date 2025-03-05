import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        """Display results based on selected visualisation"""
        viz_type = self.viz_type.get()
        viz_methods = {
            "Bar Chart": self._show_bar_chart,
            "Line Chart": self._show_line_chart,
            "Exact Values": self._show_table
        }
        
        viz_method = viz_methods.get(viz_type, self._show_bar_chart)
        viz_method(data)

    def _show_bar_chart(self, data):
        """Display grouped bar chart for model performance."""
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#F0F0F0")
        models = list(data.keys())
        metrics = list(next(iter(data.values())).keys())
        bar_width = 0.35

        # Plot each model's metrics
        for i, model in enumerate(models):
            metric_values = list(data[model].values())
            ax.bar([x + bar_width * i for x in range(len(metrics))], metric_values, bar_width, label=model, edgecolor="black", linewidth=0.7)
            
        # Axis labels
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, pad=20)

        # X-ticks and Labels
        x_tick_positions = [x + (bar_width * (len(models) / 2)) for x in range(len(metrics))]
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.legend(loc="best")
        ax.grid(True)
        fig.tight_layout()
        self._display_figure(fig)


    def _show_line_chart(self, data):
        """line chart visualisation"""
        fig, ax = plt.subplots(figsize=(8, 4))
        for model, metrics in data.items():
            ax.plot(list(metrics.values()), marker='o', label=model)
            
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics.keys(), rotation=25)
        ax.legend()
        self._display_figure(fig)
        
    def _show_table(self, data):
        """Display exact values in a table format."""
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('tight')
        ax.axis('off')

        table_data = []
        columns = ["Model"] + list(next(iter(data.values())).keys())
        for model, metrics in data.items():
            row = [model] + list(metrics.values())
            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        self._display_figure(fig)


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
