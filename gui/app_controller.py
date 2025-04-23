class AppController:
    
    """
    Controller class for the application, responsible for handling user interactions
    """
    def __init__(self, model, view): # Initialise the controller with model and view
        self.model = model
        self.view = view
        self.model.add_observer(self.view)  # Register view as an observer

# Compare the performance of selected models when user clicks the compare button
    def compare_performance(self, selected_models, perf_type="Test Set"): 
        if perf_type == "Cross-Validation":
            performance_data = self.model.get_cv_scores(selected_models)
        else:
            performance_data = self.model.get_performance(selected_models)

        self.model.notify_observers(performance_data)

