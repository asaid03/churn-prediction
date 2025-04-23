import customtkinter as ctk

class SettingsPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.pack(fill="both", expand=True)

        # Create a title label for the settings page
        title = ctk.CTkLabel(
            self,
            text="Settings",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(pady=(60, 20))

        theme_label = ctk.CTkLabel(self, text="Select Theme:", font=ctk.CTkFont(size=16))
        theme_label.pack(pady=(10, 5))

        # User can select between Light, Dark, and System themes
        self.theme = ctk.CTkOptionMenu(self, 
            values=["Light", "Dark","System"],
            command=ctk.set_appearance_mode
        )
        self.theme.set("System")
        self.theme.pack(pady=(10, 20))

        scale_label = ctk.CTkLabel(self, text="UI Scale (%):", font=ctk.CTkFont(size=16))
        scale_label.pack(pady=(10, 5))

        # Slider to adjust the UI scaling
        self.scale_slider = ctk.CTkSlider(
            self,
            from_ = 80,
            to = 120,
            number_of_steps = 8,
            command = self.scale_change,
        )
        
        self.scale_slider.set(100)
        self.scale_slider.pack(pady=(0, 20))
        
    def scale_change(self, value):
        scaling = int(value) / 100
        ctk.set_widget_scaling(scaling)