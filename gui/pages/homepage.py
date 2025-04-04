import customtkinter as ctk

class HomePage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        title = ctk.CTkLabel(
            self,
            text="Churn Prediction Dashboard",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title.pack(pady=(60, 20))

        subtitle = ctk.CTkLabel(
            self,
            text="Welcome to the Churn Prediction Dashboard",
            font=ctk.CTkFont(size=16)
        )
        subtitle.pack(pady=(0, 25))

        instruction_frame = ctk.CTkFrame(
            self,
            corner_radius=12,
            fg_color="#2B2D31",  # Dark gray
            border_width=0           
        )
        instruction_frame.pack(pady=10, padx=40, fill="both")

        instruction_text = (
            "1) Compare Metric Performance:\n"
            "   - Analyse and break down the performance metrics of different models.\n"
            "   - Compare the strengths and weaknesses of models with the aid of viual graphs.\n\n"
            "2) Settings:\n"
            "   - You can customize UI theme (Dark, Light, or System).\n"
            "   - You can adjust UI scale.\n\n"
            "3) Custom Predictions:\n"
            "   - Input your own data and get a churn prediction result."
        )

        instructions = ctk.CTkLabel(
            instruction_frame,
            text=instruction_text,
            font=ctk.CTkFont(size=14),
            justify="left",
            anchor="w",
            text_color="#FFFFFF"  # White
        )
        instructions.pack(padx=20, pady=20)

        footer = ctk.CTkLabel(
            self,
            text="Final Year Project — Churn Prediction Dashboard",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="#AAAAAA"  # Softish gray
        )
        footer.pack(side="bottom", pady=20)
