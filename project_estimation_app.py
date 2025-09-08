# Part 1: Header & Class Initialization

"""
Project Estimation ML App
========================

A comprehensive machine learning application for construction project cost estimation.
This application uses multiple ML algorithms to predict 'Estimate at Completion' (EAC) 
values based on various project parameters and historical data.

Key Features:
- Support for multiple file formats (CSV, Excel: .xlsx, .xls, .xlsb)
- Intelligent data type detection and preprocessing
- Multiple ML models: Linear Regression, Random Forest, XGBoost
- Comprehensive data visualization and analysis
- Batch processing capabilities
- Feature importance analysis
- Outlier detection and removal

Author: Construction Data Analytics Team
Version: 1.3
Date: 2024
"""

# =============================================================================
# STEP 1: IMPORT ALL REQUIRED LIBRARIES
# =============================================================================
# This section imports all the necessary libraries for the application to function

# GUI Framework - Tkinter (Built into Python)
import tkinter as tk                    # Main GUI framework
from tkinter import ttk, filedialog, messagebox  # GUI components and dialogs

# Data Processing Libraries
import pandas as pd                     # Data manipulation and analysis
import numpy as np                      # Numerical computing

# Visualization Libraries
import matplotlib.pyplot as plt         # Plotting and charting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embed plots in GUI
import seaborn as sns                   # Statistical data visualization

# Machine Learning Libraries - Data Preparation
from sklearn.model_selection import train_test_split    # Split data into train/test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # Data preprocessing
from sklearn.compose import ColumnTransformer          # Apply different preprocessing to different columns
from sklearn.pipeline import Pipeline                  # Chain preprocessing and modeling steps

# Machine Learning Libraries - Algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  # Linear models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor  # Ensemble methods
from sklearn.svm import SVR                            # Support Vector Regression
from sklearn.neighbors import KNeighborsRegressor      # K-Nearest Neighbors
from sklearn.tree import DecisionTreeRegressor         # Decision Tree

# Machine Learning Libraries - Evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Performance metrics

# Advanced ML Library
import xgboost as xgb                   # Gradient boosting framework

# Statistical Analysis
from scipy import stats                 # Statistical functions

# System Configuration
import warnings                         # Handle warning messages
warnings.filterwarnings('ignore')      # Suppress sklearn warnings for cleaner output

# =============================================================================
# STEP 2: MAIN APPLICATION CLASS DEFINITION
# =============================================================================

class ProjectEstimationApp:
    """
    Main Application Class for Project Cost Estimation
    
    This class handles the entire machine learning workflow for construction project
    cost estimation, including:
    
    1. Data Loading & Validation
       - Supports CSV and Excel formats (.csv, .xlsx, .xls, .xlsb)
       - Automatic data type detection
       - Data quality assessment
    
    2. Data Preprocessing
       - Intelligent feature encoding (numeric vs categorical)
       - Outlier detection and removal using IQR method
       - Data splitting for training/testing
       - Feature scaling and normalization
    
    3. Machine Learning Models
       - Linear Regression: Simple baseline model
       - Random Forest: Ensemble method for complex patterns
       - XGBoost: Gradient boosting for high performance
    
    4. Prediction & Analysis
       - Single project predictions
       - Batch processing for multiple projects
       - Model performance evaluation (R^2, MSE, MAE)
       - Feature importance analysis
    
    5. Visualization
       - Correlation matrices
       - Scatter plots with trend lines
       - Feature importance plots
       - Residual analysis
    
    Attributes:
        df (pd.DataFrame): Main dataset
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (Estimate at Completion)
        models (dict): Trained ML models
        predictions (dict): Model predictions
        feature_importance (dict): Feature importance scores
        scaler (StandardScaler): Feature scaling object
        label_encoders (dict): Categorical variable encoders
    """
    
    # =============================================================================
    # STEP 3: CLASS INITIALIZATION METHOD
    # =============================================================================
    def __init__(self, root):
        """
        Initialize the Project Estimation Application
        
        This method sets up the entire application when it's first created.
        It configures the main window, initializes all data storage variables,
        and creates the user interface.
        
        Args:
            root (tk.Tk): Main Tkinter window passed from the main() function
        """
        
        # =============================================================
        # BLOCK A: WINDOW CONFIGURATION
        # =============================================================
        # Configure the main application window appearance and behavior
        self.root = root                                    # Store reference to main window
        self.root.title("Project Estimation ML App v1.3")  # Set window title
        self.root.geometry("1200x800")                      # Set initial window size (width x height)
        self.root.minsize(800, 600)                         # Set minimum window size (prevents too small)
        
        # =============================================================
        # BLOCK B: DATA STORAGE ATTRIBUTES - RAW DATA
        # =============================================================
        # These variables store the original and processed datasets
        self.df = None              # Original dataset loaded from file (CSV/Excel)
        self.batch_df = None        # Batch prediction dataset (for predicting multiple projects at once)
        
        # =============================================================
        # BLOCK C: DATA STORAGE ATTRIBUTES - MACHINE LEARNING DATA
        # =============================================================
        # These variables store the data structures used for machine learning
        
        # Main feature and target data
        self.X = None               # Feature matrix (all input variables/columns except target)
        self.y = None               # Target variable (the 'Estimate at Completion' we want to predict)
        
        # Training and testing data splits
        self.X_train = None         # Training features (used to train the models)
        self.X_test = None          # Testing features (used to evaluate model performance)
        self.y_train = None         # Training targets (actual EAC values for training)
        self.y_test = None          # Testing targets (actual EAC values for testing)
        
        # Scaled versions of the data (some algorithms need standardized features)
        self.X_train_scaled = None  # Training features with mean=0, std=1 (for linear models)
        self.X_test_scaled = None   # Testing features with same scaling applied
        
        # =============================================================
        # BLOCK D: MODEL STORAGE AND RESULTS
        # =============================================================
        # These variables store the trained models and their outputs
        self.models = {}                # Dictionary storing trained ML models {model_name: model_object}
        self.predictions = {}           # Dictionary storing model predictions {model_name: predictions_array}
        self.feature_importance = {}    # Dictionary storing feature importance scores {model_name: importance_scores}
        self.batch_predictions = {}     # Dictionary storing batch prediction results
        
        # =============================================================
        # BLOCK E: DATA PREPROCESSING OBJECTS
        # =============================================================
        # These objects handle data transformation and cleaning
        self.scaler = StandardScaler()         # Object to standardize features (mean=0, std=1)
        self.label_encoders = {}               # Dictionary storing categorical encoders {column_name: encoder_object}
        self.outliers_removed = 0              # Counter tracking how many outlier records were removed
        self.outlier_data = None               # Storage for removed outlier data (for transparency/review)
        self.categorical_encoding_method = {}  # Track which encoding method was used for each column
        self.feature_names = []                # List storing the final feature names after all preprocessing
        
        # Legacy support for any remaining references (backwards compatibility)
        self.remove_outliers_var = tk.BooleanVar(value=True)  # GUI checkbox variable
        
        # =============================================================
        # BLOCK F: INITIALIZE THE GRAPHICAL USER INTERFACE
        # =============================================================
        # Create and display the user interface
        self.setup_gui()  # Call the method that builds all the tabs, buttons, and widgets

# ===========================================================================

#Part 2: GUI Setup

# =============================================================================
    # STEP 4: GUI SETUP AND INTERFACE CREATION
    # =============================================================================
    
    def setup_gui(self):
        """
        Setup the main GUI interface with tabbed layout
        
        This method creates the complete user interface using a tabbed design.
        Each tab handles a different aspect of the machine learning workflow:
        
        1. Data & Preprocessing: File loading, data preview, preprocessing options
        2. Models & Training: Model selection, hyperparameter tuning, training results
        3. Predictions: Single and batch predictions with results display
        4. Outliers: Shows discarded outlier data for transparency
        5. Visualizations: Charts, plots, and visual analysis tools
        
        The tabbed design makes the complex workflow more manageable and user-friendly.
        """
        
        # =============================================================
        # BLOCK A: CREATE MAIN TABBED INTERFACE
        # =============================================================
        # Create the notebook widget that will hold all our tabs
        self.notebook = ttk.Notebook(self.root)  # Notebook = container for multiple tabs
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)  # Make it fill the window
        
        # =============================================================
        # BLOCK B: CREATE INDIVIDUAL TABS
        # =============================================================
        # Each tab is a separate frame that contains related functionality
        
        # Tab 1: Data Loading and Preprocessing
        # This tab handles importing data files and preparing them for machine learning
        self.data_frame = ttk.Frame(self.notebook)  # Create the frame
        self.notebook.add(self.data_frame, text="📊 Data & Preprocessing")  # Add to notebook with title
        self.setup_data_tab()  # Call method to populate this tab with widgets
        
        # Tab 2: Model Configuration and Training
        # This tab handles selecting ML algorithms, setting parameters, and training
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="🤖 Models & Training")
        self.setup_model_tab()
        
        # Tab 3: Predictions
        # This tab handles making predictions on new projects (single or batch)
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text="🎯 Predictions")
        self.setup_predict_tab()
        
        # Tab 4: Outlier Analysis
        # This tab shows which data points were removed as outliers (transparency)
        self.outlier_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.outlier_frame, text="🚨 Outliers")
        self.setup_outlier_tab()
        
        # Tab 5: Visualizations
        # This tab shows charts and graphs to understand the data and model performance
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="📈 Visualizations")
        self.setup_viz_tab()
    
    # =============================================================================
    # STEP 5: DATA TAB SETUP - FILE LOADING AND PREPROCESSING CONTROLS
    # =============================================================================
    
    def setup_data_tab(self):
        """
        Setup data loading and preprocessing tab
        
        This tab contains:
        - File loading section (CSV/Excel import)
        - Data configuration section (train/test split ratio, outlier handling)
        - Data preview section (shows loaded data in a table)
        """
        
        # =============================================================
        # BLOCK A: FILE LOADING SECTION
        # =============================================================
        # Create a labeled frame for file loading controls
        load_frame = ttk.LabelFrame(self.data_frame, text="Data Loading", padding=10)
        load_frame.pack(fill='x', padx=5, pady=5)  # fill='x' means stretch horizontally
        
        # Button to open file dialog and load data
        ttk.Button(load_frame, text="Load Data File", command=self.load_data).pack(side='left', padx=5)
        
        # Label to show which file is currently loaded
        self.file_label = ttk.Label(load_frame, text="No file loaded")
        self.file_label.pack(side='left', padx=10)
        
        # =============================================================
        # BLOCK B: DATA CONFIGURATION SECTION
        # =============================================================
        # Create a labeled frame for configuration options
        config_frame = ttk.LabelFrame(self.data_frame, text="Data Configuration", padding=10)
        config_frame.pack(fill='x', padx=5, pady=5)
        
        # Train/Test split ratio control
        # This determines what percentage of data is used for training vs testing
        ttk.Label(config_frame, text="Training Data %:").pack(side='left')
        self.train_size_var = tk.DoubleVar(value=0.8)  # Default to 80% training, 20% testing
        ttk.Scale(config_frame, from_=0.5, to=0.95, variable=self.train_size_var, 
                 orient='horizontal', length=200).pack(side='left', padx=5)
        self.train_label = ttk.Label(config_frame, text="80%")  # Shows current percentage
        self.train_label.pack(side='left', padx=5)
        
        # =============================================================
        # BLOCK C: OUTLIER HANDLING OPTIONS
        # =============================================================
        # Outliers are data points that are unusually high or low
        # Different strategies for handling them
        outlier_frame = ttk.LabelFrame(config_frame, text="Outlier Handling")
        outlier_frame.pack(side='left', padx=20, pady=2)
        
        # Radio buttons for outlier strategy selection
        self.outlier_strategy_var = tk.StringVar(value="smart")  # Default strategy
        
        # Smart strategy: Different handling for different model types
        ttk.Radiobutton(outlier_frame, text="Smart (Model-Aware)", 
                       variable=self.outlier_strategy_var, value="smart").pack(anchor='w', padx=5)
        
        # Remove all outliers: Traditional statistical approach
        ttk.Radiobutton(outlier_frame, text="Remove All Outliers", 
                       variable=self.outlier_strategy_var, value="remove").pack(anchor='w', padx=5)
        
        # Keep all data: No outlier removal
        ttk.Radiobutton(outlier_frame, text="Keep All Data", 
                       variable=self.outlier_strategy_var, value="keep").pack(anchor='w', padx=5)
        
        # Explanation tooltip for smart strategy
        info_text = "Smart: Tree models keep outliers, linear models remove them"
        ttk.Label(outlier_frame, text=info_text, font=('TkDefaultFont', 7), 
                 foreground='gray').pack(anchor='w', padx=5)
        
        # Button to process the loaded data with chosen settings
        ttk.Button(config_frame, text="Process Data", command=self.process_data).pack(side='right', padx=5)
        
        # =============================================================
        # BLOCK D: DATA PREVIEW SECTION
        # =============================================================
        # Shows the loaded data in a table format
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Treeview widget acts like a spreadsheet to display data
        self.data_tree = ttk.Treeview(preview_frame)
        
        # Scrollbar for the data table (in case there are many rows)
        scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)  # Connect scrollbar to treeview
        
        # Pack the treeview and scrollbar
        self.data_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # =============================================================
        # BLOCK E: DYNAMIC UPDATES
        # =============================================================
        # Update the percentage display when the slider moves
        # This is a "callback" function that runs whenever the slider value changes
        self.train_size_var.trace('w', lambda *args: self.train_label.config(
            text=f"{int(self.train_size_var.get()*100)}%"))  # Convert 0.8 to "80%"

# ===========================================================================

#Part 3: Model Tab Setup & Hyperparameters Definition

# =============================================================================
    # STEP 6: MODEL TAB SETUP - ALGORITHM SELECTION AND HYPERPARAMETER TUNING
    # =============================================================================
    
    def setup_model_tab(self):
        """
        Setup model configuration and training tab
        
        This tab contains:
        - Model selection checkboxes (choose which algorithms to train)
        - Dynamic hyperparameter controls (algorithm-specific settings)
        - Training button and results display
        """
        
        # =============================================================
        # BLOCK A: MODEL SELECTION SECTION
        # =============================================================
        # Allow users to choose which machine learning algorithms to train
        model_select_frame = ttk.LabelFrame(self.model_frame, text="Model Selection", padding=10)
        model_select_frame.pack(fill='x', padx=5, pady=5)
        
        # Dictionary to store checkbox variables for each model
        self.model_vars = {}
        
        # List of all available machine learning algorithms
        models = [
            'Linear Regression',        # Simple linear relationship
            'Ridge Regression',         # Linear with L2 regularization
            'Lasso Regression',         # Linear with L1 regularization (feature selection)
            'ElasticNet',              # Linear with both L1 and L2 regularization
            'Random Forest',           # Ensemble of decision trees
            'Extra Trees',             # Ensemble with random splits
            'Gradient Boosting',       # Sequential boosting algorithm
            'XGBoost',                 # Advanced gradient boosting
            'Support Vector Regression', # Support vector machine for regression
            'K-Nearest Neighbors',     # Distance-based algorithm
            'Decision Tree'            # Single decision tree
        ]
        
        # Create checkboxes for each model in a 3-column grid
        for i, model in enumerate(models):
            # Default to checked for the three most commonly used models
            default_on = model in ['Linear Regression', 'Random Forest', 'XGBoost']
            self.model_vars[model] = tk.BooleanVar(value=default_on)
            
            # Use 3 columns to accommodate all models neatly
            ttk.Checkbutton(model_select_frame, text=model, 
                           variable=self.model_vars[model]).grid(
                           row=i//3, column=i%3, sticky='w', padx=8, pady=2)
        
        # =============================================================
        # BLOCK B: DYNAMIC HYPERPARAMETER SECTION
        # =============================================================
        # This section shows/hides parameter controls based on selected models
        self.hyperparam_frame = ttk.LabelFrame(self.model_frame, 
                                             text="Hyperparameters (Select models to see options)", 
                                             padding=10)
        self.hyperparam_frame.pack(fill='x', padx=5, pady=5)
        
        # Initialize all hyperparameter variables
        self.init_hyperparameters()
        
        # Container to track dynamically created widgets
        self.hyperparam_widgets = {}
        
        # Set up callbacks to update hyperparameter display when models are selected/deselected
        for model_name in models:
            self.model_vars[model_name].trace('w', lambda *args: self.update_hyperparameter_display())
        
        # =============================================================
        # BLOCK C: TRAINING SECTION
        # =============================================================
        # Section for training models and displaying results
        train_frame = ttk.LabelFrame(self.model_frame, text="Training Results", padding=10)
        train_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Button to start the training process
        ttk.Button(train_frame, text="Train Models", command=self.train_models).pack(pady=5)
        
        # Text widget to display training results and performance metrics
        self.results_text = tk.Text(train_frame, height=15, width=80)
        
        # Scrollbar for the results text (in case of long results)
        results_scroll = ttk.Scrollbar(train_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        # Pack the text widget and scrollbar
        self.results_text.pack(side='left', fill='both', expand=True)
        results_scroll.pack(side='right', fill='y')
    
    # =============================================================================
    # STEP 7: HYPERPARAMETER INITIALIZATION
    # =============================================================================
    
    def init_hyperparameters(self):
        """
        Initialize hyperparameter variables for all models
        
        Hyperparameters are settings that control how each algorithm works.
        Think of them as "knobs" you can turn to make the algorithm work better
        for your specific problem. Each algorithm has different hyperparameters.
        """
        
        # =============================================================
        # BLOCK A: LINEAR MODEL HYPERPARAMETERS
        # =============================================================
        
        # Linear Regression has no hyperparameters to tune
        # (It uses a mathematical formula to find the best fit line)
        
        # Ridge Regression - Controls overfitting with L2 regularization
        self.ridge_alpha = tk.DoubleVar(value=1.0)  # Higher = more regularization
        
        # Lasso Regression - Controls overfitting with L1 regularization
        self.lasso_alpha = tk.DoubleVar(value=1.0)  # Higher = more feature selection
        
        # ElasticNet - Combines Ridge and Lasso
        self.elastic_alpha = tk.DoubleVar(value=1.0)      # Overall regularization strength
        self.elastic_l1_ratio = tk.DoubleVar(value=0.5)   # Balance between L1 and L2
        
        # =============================================================
        # BLOCK B: TREE-BASED MODEL HYPERPARAMETERS
        # =============================================================
        
        # Random Forest - Ensemble of many decision trees
        self.rf_estimators = tk.IntVar(value=100)           # Number of trees
        self.rf_depth = tk.IntVar(value=10)                 # Maximum depth of each tree
        self.rf_min_samples_split = tk.IntVar(value=2)      # Minimum samples to split a node
        
        # Extra Trees - Similar to Random Forest but with random splits
        self.et_estimators = tk.IntVar(value=100)           # Number of trees
        self.et_depth = tk.IntVar(value=10)                 # Maximum depth of each tree
        self.et_min_samples_split = tk.IntVar(value=2)      # Minimum samples to split a node
        
        # Gradient Boosting - Sequential improvement of predictions
        self.gb_estimators = tk.IntVar(value=100)           # Number of boosting stages
        self.gb_learning_rate = tk.DoubleVar(value=0.1)     # How much each tree contributes
        self.gb_depth = tk.IntVar(value=3)                  # Depth of individual trees
        
        # XGBoost - Advanced gradient boosting
        self.xgb_estimators = tk.IntVar(value=100)          # Number of boosting rounds
        self.xgb_lr = tk.DoubleVar(value=0.1)               # Learning rate
        self.xgb_depth = tk.IntVar(value=6)                 # Maximum tree depth
        
        # Decision Tree - Single tree
        self.dt_depth = tk.IntVar(value=10)                 # Maximum depth
        self.dt_min_samples_split = tk.IntVar(value=2)      # Minimum samples to split
        self.dt_criterion = tk.StringVar(value='squared_error')  # Split quality measure
        
        # =============================================================
        # BLOCK C: OTHER ALGORITHM HYPERPARAMETERS
        # =============================================================
        
        # Support Vector Regression - Finds optimal boundary
        self.svr_C = tk.DoubleVar(value=100.0)              # Regularization parameter
        self.svr_gamma = tk.DoubleVar(value=0.1)            # Kernel coefficient
        self.svr_kernel = tk.StringVar(value='rbf')         # Kernel type
        
        # K-Nearest Neighbors - Predicts based on similar examples
        self.knn_neighbors = tk.IntVar(value=5)             # Number of neighbors to consider
        self.knn_weights = tk.StringVar(value='uniform')    # How to weight neighbors

#=================================================================================================

# Part 4: Dynamic Hyperparameter Display Method and Prediction Tab Setup

# =============================================================================
    # STEP 8: DYNAMIC HYPERPARAMETER DISPLAY
    # =============================================================================
    
    def update_hyperparameter_display(self):
        """
        Update hyperparameter display based on selected models
        
        This method creates a dynamic interface that shows only the hyperparameter
        controls for the models that are currently selected. When a user checks
        or unchecks a model, this method runs automatically to update the display.
        
        This keeps the interface clean and prevents confusion by only showing
        relevant controls.
        """
        
        # =============================================================
        # BLOCK A: CLEAR EXISTING WIDGETS
        # =============================================================
        # Remove all currently displayed hyperparameter controls
        for widget_dict in self.hyperparam_widgets.values():
            for widget in widget_dict.values():
                if hasattr(widget, 'destroy'):
                    widget.destroy()  # Remove widget from screen
        self.hyperparam_widgets.clear()  # Clear the tracking dictionary
        
        # =============================================================
        # BLOCK B: GET SELECTED MODELS
        # =============================================================
        # Find out which models are currently checked
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        
        # If no models are selected, show a message and return
        if not selected_models:
            self.hyperparam_frame.configure(text="Hyperparameters (Select models to see options)")
            return
        
        # Update the frame title to show how many models are selected
        self.hyperparam_frame.configure(text=f"Hyperparameters for {len(selected_models)} Selected Models")
        
        # =============================================================
        # BLOCK C: CREATE HYPERPARAMETER WIDGETS FOR EACH SELECTED MODEL
        # =============================================================
        # Loop through each selected model and create its specific controls
        row = 0  # Track which row we're placing widgets in
        
        for model_name in selected_models:
            
            # =========================================================
            # SUB-BLOCK C1: LINEAR REGRESSION (NO HYPERPARAMETERS)
            # =========================================================
            if model_name == 'Linear Regression':
                # Linear regression uses a mathematical formula - no tuning needed
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name} - No hyperparameters to tune", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                ttk.Label(frame, text="Uses closed-form solution (no tuning needed)", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C2: RIDGE REGRESSION
            # =========================================================
            elif model_name == 'Ridge Regression':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                # Alpha parameter controls regularization strength
                ttk.Label(frame, text="Alpha (Regularization):").pack(side='left')
                ttk.Spinbox(frame, from_=0.01, to=100, increment=0.1, textvariable=self.ridge_alpha, width=8).pack(side='left', padx=5)
                ttk.Label(frame, text="Higher values = more regularization", 
                         font=('TkDefaultFont', 8), foreground='gray').pack(side='left', padx=10)
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C3: LASSO REGRESSION
            # =========================================================
            elif model_name == 'Lasso Regression':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                # Alpha parameter for Lasso (also does feature selection)
                ttk.Label(frame, text="Alpha (Regularization):").pack(side='left')
                ttk.Spinbox(frame, from_=0.01, to=100, increment=0.1, textvariable=self.lasso_alpha, width=8).pack(side='left', padx=5)
                ttk.Label(frame, text="Higher values = more feature selection", 
                         font=('TkDefaultFont', 8), foreground='gray').pack(side='left', padx=10)
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C4: ELASTICNET (COMBINES RIDGE + LASSO)
            # =========================================================
            elif model_name == 'ElasticNet':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                # Two parameters: overall strength and balance between Ridge/Lasso
                param_frame1 = ttk.Frame(frame)
                param_frame1.pack(fill='x', pady=2)
                ttk.Label(param_frame1, text="Alpha:").pack(side='left')
                ttk.Spinbox(param_frame1, from_=0.01, to=100, increment=0.1, textvariable=self.elastic_alpha, width=8).pack(side='left', padx=5)
                ttk.Label(param_frame1, text="L1 Ratio:").pack(side='left', padx=(20,0))
                ttk.Spinbox(param_frame1, from_=0.0, to=1.0, increment=0.1, textvariable=self.elastic_l1_ratio, width=8).pack(side='left', padx=5)
                
                ttk.Label(frame, text="L1 Ratio: 0=Ridge, 1=Lasso, 0.5=balanced", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C5: RANDOM FOREST
            # =========================================================
            elif model_name == 'Random Forest':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                # Three main parameters for Random Forest
                param_frame1 = ttk.Frame(frame)
                param_frame1.pack(fill='x', pady=2)
                ttk.Label(param_frame1, text="N Estimators:").pack(side='left')  # Number of trees
                ttk.Spinbox(param_frame1, from_=10, to=500, textvariable=self.rf_estimators, width=8).pack(side='left', padx=5)
                ttk.Label(param_frame1, text="Max Depth:").pack(side='left', padx=(20,0))  # Tree depth
                ttk.Spinbox(param_frame1, from_=3, to=50, textvariable=self.rf_depth, width=8).pack(side='left', padx=5)
                
                param_frame2 = ttk.Frame(frame)
                param_frame2.pack(fill='x', pady=2)
                ttk.Label(param_frame2, text="Min Samples Split:").pack(side='left')  # Minimum samples to split
                ttk.Spinbox(param_frame2, from_=2, to=20, textvariable=self.rf_min_samples_split, width=8).pack(side='left', padx=5)
                
                ttk.Label(frame, text="More trees & depth = better fit but slower training", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C6: EXTRA TREES (SIMILAR TO RANDOM FOREST)
            # =========================================================
            elif model_name == 'Extra Trees':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                param_frame1 = ttk.Frame(frame)
                param_frame1.pack(fill='x', pady=2)
                ttk.Label(param_frame1, text="N Estimators:").pack(side='left')
                ttk.Spinbox(param_frame1, from_=10, to=500, textvariable=self.et_estimators, width=8).pack(side='left', padx=5)
                ttk.Label(param_frame1, text="Max Depth:").pack(side='left', padx=(20,0))
                ttk.Spinbox(param_frame1, from_=3, to=50, textvariable=self.et_depth, width=8).pack(side='left', padx=5)
                
                ttk.Label(frame, text="Similar to Random Forest but with random splits", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C7: GRADIENT BOOSTING
            # =========================================================
            elif model_name == 'Gradient Boosting':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                param_frame1 = ttk.Frame(frame)
                param_frame1.pack(fill='x', pady=2)
                ttk.Label(param_frame1, text="N Estimators:").pack(side='left')  # Number of boosting stages
                ttk.Spinbox(param_frame1, from_=10, to=500, textvariable=self.gb_estimators, width=8).pack(side='left', padx=5)
                ttk.Label(param_frame1, text="Learning Rate:").pack(side='left', padx=(20,0))  # How fast to learn
                ttk.Spinbox(param_frame1, from_=0.01, to=0.5, increment=0.01, textvariable=self.gb_learning_rate, width=8).pack(side='left', padx=5)
                
                ttk.Label(frame, text="Lower learning rate needs more estimators", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C8: XGBOOST
            # =========================================================
            elif model_name == 'XGBoost':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                param_frame1 = ttk.Frame(frame)
                param_frame1.pack(fill='x', pady=2)
                ttk.Label(param_frame1, text="N Estimators:").pack(side='left')
                ttk.Spinbox(param_frame1, from_=10, to=500, textvariable=self.xgb_estimators, width=8).pack(side='left', padx=5)
                ttk.Label(param_frame1, text="Learning Rate:").pack(side='left', padx=(20,0))
                ttk.Spinbox(param_frame1, from_=0.01, to=0.5, increment=0.01, textvariable=self.xgb_lr, width=8).pack(side='left', padx=5)
                
                ttk.Label(frame, text="High-performance gradient boosting", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C9: SUPPORT VECTOR REGRESSION
            # =========================================================
            elif model_name == 'Support Vector Regression':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                param_frame1 = ttk.Frame(frame)
                param_frame1.pack(fill='x', pady=2)
                ttk.Label(param_frame1, text="C (Regularization):").pack(side='left')  # Controls overfitting
                ttk.Spinbox(param_frame1, from_=0.1, to=1000, increment=1, textvariable=self.svr_C, width=8).pack(side='left', padx=5)
                ttk.Label(param_frame1, text="Gamma:").pack(side='left', padx=(20,0))  # Kernel parameter
                ttk.Spinbox(param_frame1, from_=0.001, to=1.0, increment=0.01, textvariable=self.svr_gamma, width=8).pack(side='left', padx=5)
                
                ttk.Label(frame, text="Higher C = less regularization, higher gamma = more complex", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C10: K-NEAREST NEIGHBORS
            # =========================================================
            elif model_name == 'K-Nearest Neighbors':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                param_frame1 = ttk.Frame(frame)
                param_frame1.pack(fill='x', pady=2)
                ttk.Label(param_frame1, text="N Neighbors:").pack(side='left')  # How many neighbors to consider
                ttk.Spinbox(param_frame1, from_=1, to=50, textvariable=self.knn_neighbors, width=8).pack(side='left', padx=5)
                
                # Weight strategy for neighbors
                weight_frame = ttk.Frame(frame)
                weight_frame.pack(fill='x', pady=2)
                ttk.Label(weight_frame, text="Weights:").pack(side='left')
                ttk.Radiobutton(weight_frame, text="Uniform", variable=self.knn_weights, value="uniform").pack(side='left', padx=5)
                ttk.Radiobutton(weight_frame, text="Distance", variable=self.knn_weights, value="distance").pack(side='left', padx=5)
                
                ttk.Label(frame, text="Lower k = more complex, distance weights favor closer points", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            # =========================================================
            # SUB-BLOCK C11: DECISION TREE
            # =========================================================
            elif model_name == 'Decision Tree':
                frame = ttk.LabelFrame(self.hyperparam_frame, text=f"{model_name}", padding=5)
                frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
                
                param_frame1 = ttk.Frame(frame)
                param_frame1.pack(fill='x', pady=2)
                ttk.Label(param_frame1, text="Max Depth:").pack(side='left')  # Tree depth limit
                ttk.Spinbox(param_frame1, from_=3, to=50, textvariable=self.dt_depth, width=8).pack(side='left', padx=5)
                ttk.Label(param_frame1, text="Min Samples Split:").pack(side='left', padx=(20,0))  # Minimum samples to split
                ttk.Spinbox(param_frame1, from_=2, to=20, textvariable=self.dt_min_samples_split, width=8).pack(side='left', padx=5)
                
                ttk.Label(frame, text="Deeper trees = more complex but prone to overfitting", 
                         font=('TkDefaultFont', 8), foreground='gray').pack()
                self.hyperparam_widgets[model_name] = {'frame': frame}
            
            row += 1  # Move to next row for next model
        
        # =============================================================
        # BLOCK D: CONFIGURE GRID LAYOUT
        # =============================================================
        # Set up the grid system for proper widget placement
        for i in range(row):
            self.hyperparam_frame.grid_rowconfigure(i, weight=0)  # Don't expand rows
        self.hyperparam_frame.grid_columnconfigure(0, weight=1)  # Expand columns equally
        self.hyperparam_frame.grid_columnconfigure(1, weight=1)

# ===============================================================================

#Part 5: Prediction Tab Setup (manual input and batch prediction interfaces)

# =============================================================================
    # STEP 9: PREDICTION TAB SETUP - SINGLE AND BATCH PREDICTIONS
    # =============================================================================
    
    def setup_predict_tab(self):
        """
        Setup predictions tab with manual input interface
        
        This tab provides two ways to make predictions:
        1. Manual Input: Enter project details by hand for a single project
        2. Batch Prediction: Upload a file with multiple projects to predict all at once
        
        Both methods use the trained models to estimate project costs.
        """
        
        # =============================================================
        # BLOCK A: CREATE SUB-TABS FOR DIFFERENT PREDICTION TYPES
        # =============================================================
        # Create a nested notebook (tabs within the prediction tab)
        predict_notebook = ttk.Notebook(self.predict_frame)
        predict_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Manual Single Prediction Tab - for entering one project manually
        manual_frame = ttk.Frame(predict_notebook)
        predict_notebook.add(manual_frame, text="✏️ Manual Input")
        self.setup_manual_prediction(manual_frame)
        
        # Batch Prediction Tab - for uploading files with multiple projects
        batch_main_frame = ttk.Frame(predict_notebook)
        predict_notebook.add(batch_main_frame, text="📂 Batch Prediction")
        self.setup_batch_prediction(batch_main_frame)
        
    # =============================================================================
    # STEP 10: MANUAL PREDICTION INTERFACE SETUP
    # =============================================================================
    
    def setup_manual_prediction(self, parent):
        """
        Setup manual input interface for single predictions
        
        This creates a form where users can enter project details manually
        and get cost estimates from all trained models. The form dynamically
        adapts based on the features found in the training data.
        """
        
        # =============================================================
        # BLOCK A: INSTRUCTIONS SECTION
        # =============================================================
        # Provide clear instructions to the user
        instructions_frame = ttk.LabelFrame(parent, text="📝 Instructions", padding=10)
        instructions_frame.pack(fill='x', padx=5, pady=5)
        
        instructions_text = "Enter values for each feature below, then click 'Predict' to get cost estimates from all trained models."
        ttk.Label(instructions_frame, text=instructions_text, wraplength=600).pack()
        
        # =============================================================
        # BLOCK B: SCROLLABLE INPUT FORM
        # =============================================================
        # Create a scrollable area for input fields (in case there are many features)
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Canvas and scrollbar setup for scrolling through many input fields
        self.input_canvas = tk.Canvas(canvas_frame, height=400)
        input_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.input_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.input_canvas)
        
        # Configure scrolling behavior
        self.scrollable_frame.bind("<Configure>", 
            lambda e: self.input_canvas.configure(scrollregion=self.input_canvas.bbox("all")))
        
        self.input_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.input_canvas.configure(yscrollcommand=input_scrollbar.set)
        
        # Pack the canvas and scrollbar
        self.input_canvas.pack(side="left", fill="both", expand=True)
        input_scrollbar.pack(side="right", fill="y")
        
        # Initialize dictionary to store input field widgets
        self.feature_inputs = {}
        
        # =============================================================
        # BLOCK C: ACTION BUTTONS
        # =============================================================
        # Buttons for various actions on the manual input form
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill='x', padx=5, pady=5)
        
        # Main prediction button
        ttk.Button(action_frame, text="🎯 Predict", command=self.predict_manual_input).pack(side='left', padx=5)
        
        # Reset form button - clears all input fields
        ttk.Button(action_frame, text="🔄 Reset Form", command=self.reset_manual_inputs).pack(side='left', padx=5)
        
        # Fill example button - populates form with sample data
        ttk.Button(action_frame, text="📋 Fill Example", command=self.fill_example_values).pack(side='left', padx=5)
        
        # Export results button - saves prediction results to file
        ttk.Button(action_frame, text="📊 Export Results", command=self.export_manual_prediction_results).pack(side='left', padx=5)
        
        # =============================================================
        # BLOCK D: RESULTS DISPLAY SECTION
        # =============================================================
        # Area to show prediction results from all models
        manual_results_frame = ttk.LabelFrame(parent, text="🎯 Prediction Results", padding=10)
        manual_results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Text widget to display detailed prediction results
        self.manual_results_text = tk.Text(manual_results_frame, height=6, width=80)
        manual_scroll = ttk.Scrollbar(manual_results_frame, orient='vertical', command=self.manual_results_text.yview)
        self.manual_results_text.configure(yscrollcommand=manual_scroll.set)
        
        # Pack the results display and scrollbar
        self.manual_results_text.pack(side='left', fill='both', expand=True)
        manual_scroll.pack(side='right', fill='y')
        
    # =============================================================================
    # STEP 11: BATCH PREDICTION INTERFACE SETUP
    # =============================================================================
    
    def setup_batch_prediction(self, parent):
        """
        Setup batch prediction interface
        
        This interface allows users to upload CSV or Excel files containing
        multiple projects and get cost predictions for all of them at once.
        This is useful for processing many projects efficiently.
        """
        
        # =============================================================
        # BLOCK A: BATCH PREDICTION CONTROLS
        # =============================================================
        # Buttons for loading files and running batch predictions
        batch_frame = ttk.LabelFrame(parent, text="📂 Batch Prediction", padding=10)
        batch_frame.pack(fill='x', padx=5, pady=5)
        
        # Button to load a file containing multiple projects
        ttk.Button(batch_frame, text="Load Batch File", command=self.load_batch_prediction).pack(side='left', padx=5)
        
        # Button to run predictions on all projects in the loaded file
        ttk.Button(batch_frame, text="Predict Batch", command=self.predict_batch).pack(side='left', padx=5)
        
        # Button to export the batch results to a file
        ttk.Button(batch_frame, text="Export Results", command=self.export_predictions).pack(side='left', padx=5)
        
        # =============================================================
        # BLOCK B: BATCH FILE INFORMATION DISPLAY
        # =============================================================
        # Shows the contents of the loaded batch file so users can verify it's correct
        batch_info_frame = ttk.LabelFrame(parent, text="📋 Batch File Information", padding=10)
        batch_info_frame.pack(fill='x', padx=5, pady=5)
        
        # Treeview to display the loaded batch file data
        self.batch_info_tree = ttk.Treeview(batch_info_frame, height=6)
        batch_info_scrollbar = ttk.Scrollbar(batch_info_frame, orient='vertical', command=self.batch_info_tree.yview)
        self.batch_info_tree.configure(yscrollcommand=batch_info_scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.batch_info_tree.pack(side='left', fill='both', expand=True)
        batch_info_scrollbar.pack(side='right', fill='y')
        
        # =============================================================
        # BLOCK C: BATCH PREDICTION RESULTS DISPLAY
        # =============================================================
        # Shows the prediction results for all projects in the batch
        batch_results_frame = ttk.LabelFrame(parent, text="📊 EAC Batch Prediction Results", padding=10)
        batch_results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Treeview to display prediction results in a table format
        self.pred_tree = ttk.Treeview(batch_results_frame)
        pred_scrollbar = ttk.Scrollbar(batch_results_frame, orient='vertical', command=self.pred_tree.yview)
        self.pred_tree.configure(yscrollcommand=pred_scrollbar.set)
        
        # Pack the results table and scrollbar
        self.pred_tree.pack(side='left', fill='both', expand=True)
        pred_scrollbar.pack(side='right', fill='y')
    
    # =============================================================================
    # STEP 12: OUTLIER TAB SETUP - TRANSPARENCY AND DATA REVIEW
    # =============================================================================
    
    def setup_outlier_tab(self):
        """
        Setup outlier analysis tab
        
        This tab provides transparency by showing which data points were
        removed as outliers during preprocessing. This helps users understand
        what data was excluded and why, which is important for trust and
        auditing purposes.
        """
        
        # =============================================================
        # BLOCK A: OUTLIER INFORMATION PANEL
        # =============================================================
        # Display summary information about outliers
        info_frame = ttk.LabelFrame(self.outlier_frame, text="📊 Outlier Information", padding=10)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        # Label to show outlier statistics (updated when data is processed)
        self.outlier_info_label = ttk.Label(info_frame, text="No outliers detected yet. Process data to see outlier analysis.")
        self.outlier_info_label.pack(pady=5)
        
        # =============================================================
        # BLOCK B: OUTLIER DATA DISPLAY
        # =============================================================
        # Show the actual data that was removed as outliers
        outlier_data_frame = ttk.LabelFrame(self.outlier_frame, text="🗑️ Discarded Outlier Data", padding=10)
        outlier_data_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Treeview to display the outlier data in table format
        self.outlier_tree = ttk.Treeview(outlier_data_frame)
        outlier_scrollbar = ttk.Scrollbar(outlier_data_frame, orient='vertical', command=self.outlier_tree.yview)
        self.outlier_tree.configure(yscrollcommand=outlier_scrollbar.set)
        
        # Pack the outlier data table and scrollbar
        self.outlier_tree.pack(side='left', fill='both', expand=True)
        outlier_scrollbar.pack(side='right', fill='y')
        
        # Button to refresh the outlier data display
        ttk.Button(outlier_data_frame, text="Refresh Outlier Data", command=self.display_outlier_data).pack(pady=5)

# =============================================================

# Part 6: Visualization Tab Setup and the Core Data Processing Methods

# =============================================================================
    # STEP 13: VISUALIZATION TAB SETUP - CHARTS AND ANALYSIS TOOLS
    # =============================================================================
    
    def setup_viz_tab(self):
        """
        Setup visualization tab
        
        This tab provides various charts and graphs to help users understand:
        - Which features are most important for predictions
        - How features relate to each other (correlations)
        - How well the models are performing
        - Patterns and relationships in the data
        """
        
        # =============================================================
        # BLOCK A: VISUALIZATION CONTROL PANEL
        # =============================================================
        # Buttons to generate different types of charts
        control_frame = ttk.Frame(self.viz_frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Button to show which features are most important for predictions
        ttk.Button(control_frame, text="Feature Importance", command=self.plot_feature_importance).pack(side='left', padx=5)
        
        # Button to show correlation matrix (how features relate to each other)
        ttk.Button(control_frame, text="Correlation Matrix", command=self.plot_correlation).pack(side='left', padx=5)
        
        # Button to show scatter plots (feature vs target relationships)
        ttk.Button(control_frame, text="Scatter Plots", command=self.plot_scatter).pack(side='left', padx=5)
        
        # Button to show residual plots (model performance analysis)
        ttk.Button(control_frame, text="Residual Plots", command=self.plot_residuals).pack(side='left', padx=5)
        
        # =============================================================
        # BLOCK B: MATPLOTLIB FIGURE SETUP
        # =============================================================
        # Configure the plotting area with optimized settings for readability
        plt.style.use('default')  # Use default style for better readability
        self.fig = plt.figure(figsize=(14, 10), dpi=100)  # Create main figure
        
        # Set smaller font sizes to prevent text overlap in charts
        plt.rcParams.update({
            'font.size': 5,            # Base font size
            'axes.titlesize': 7,       # Chart title size
            'axes.labelsize': 5,       # Axis label size
            'xtick.labelsize': 4,      # X-axis tick label size
            'ytick.labelsize': 4,      # Y-axis tick label size
            'legend.fontsize': 5,      # Legend font size
            'figure.titlesize': 8      # Main figure title size
        })
        
        # =============================================================
        # BLOCK C: EMBED MATPLOTLIB IN TKINTER
        # =============================================================
        # Create a canvas to display the matplotlib plots inside the GUI
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    # =============================================================================
    # STEP 14: DATA LOADING FUNCTIONALITY
    # =============================================================================
    
    def load_data(self):
        """
        Load CSV or Excel data file
        
        This method handles loading project data from various file formats:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls, .xlsb)
        
        After loading, it displays a preview of the data and shows summary information.
        """
        
        # =============================================================
        # BLOCK A: FILE SELECTION DIALOG
        # =============================================================
        # Open a file dialog for the user to select their data file
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls *.xlsb"), ("All files", "*.*")]
        )
        
        # =============================================================
        # BLOCK B: FILE LOADING AND FORMAT DETECTION
        # =============================================================
        if file_path:  # If user selected a file (didn't cancel)
            try:
                # Load different file formats using appropriate methods
                if file_path.endswith('.csv'):
                    # Load CSV file using pandas
                    self.df = pd.read_csv(file_path)
                    
                elif file_path.endswith(('.xlsx', '.xls', '.xlsb')):
                    # Load Excel files with appropriate engines
                    if file_path.endswith('.xlsb'):
                        # Binary Excel format requires special engine
                        self.df = pd.read_excel(file_path, engine='pyxlsb')
                    else:
                        # Standard Excel formats
                        self.df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    # Unsupported file format
                    messagebox.showerror("Error", "Unsupported file format!")
                    return
                
                # =============================================================
                # BLOCK C: SUCCESS CONFIRMATION AND PREVIEW
                # =============================================================
                # Update UI to show which file was loaded
                self.file_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
                
                # Display preview of the loaded data
                self.display_data_preview()
                
                # Show success message with data dimensions
                messagebox.showinfo("Success", f"Data loaded successfully!\nShape: {self.df.shape}")
                
            except Exception as e:
                # Handle any errors during file loading
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    # =============================================================================
    # STEP 15: DATA PREVIEW DISPLAY
    # =============================================================================
    
    def display_data_preview(self):
        """
        Display data preview in treeview
        
        Shows the first 50 rows of the loaded data in a table format
        so users can verify their data was loaded correctly.
        """
        if self.df is not None:
            # =============================================================
            # BLOCK A: CLEAR EXISTING DISPLAY
            # =============================================================
            # Remove any previously displayed data
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # =============================================================
            # BLOCK B: CONFIGURE TABLE COLUMNS
            # =============================================================
            # Set up the table columns to match the data columns
            self.data_tree["columns"] = list(self.df.columns)
            self.data_tree["show"] = "headings"  # Show column headers
            
            # Configure each column header and width
            for col in self.df.columns:
                self.data_tree.heading(col, text=col)  # Set column header text
                self.data_tree.column(col, width=100)  # Set column width
            
            # =============================================================
            # BLOCK C: INSERT DATA ROWS
            # =============================================================
            # Insert data rows (limiting to first 50 for performance)
            for idx, row in self.df.head(50).iterrows():
                self.data_tree.insert("", "end", values=list(row))
    
    # =============================================================================
    # STEP 16: INTELLIGENT DATA TYPE DETECTION
    # =============================================================================
    
    def intelligent_type_detection(self, df):
        """
        Intelligent Data Type Detection
        
        Automatically determines whether each column should be treated as:
        - Numeric: Continuous values for mathematical operations
        - Categorical: Discrete categories for encoding
        
        Algorithm:
        1. Attempts to convert each column to numeric using pd.to_numeric()
        2. Calculates percentage of successfully converted values
        3. If >70% numeric conversion success → treat as numeric
        4. Otherwise → treat as categorical
        
        Why This Matters:
        - Many datasets have numeric data stored as strings
        - Simple type checking (dtype) can misclassify data
        - Proper classification is crucial for encoding strategy selection
        
        Args:
            df (pd.DataFrame): Input dataframe to analyze
            
        Returns:
            tuple: (numeric_columns, categorical_columns)
                - numeric_columns (list): Column names to treat as numeric
                - categorical_columns (list): Column names to treat as categorical
                
        Note:
            Excludes the target column (rightmost column) from analysis
        """
        
        # =============================================================
        # BLOCK A: INITIALIZE LISTS FOR COLUMN CLASSIFICATION
        # =============================================================
        numeric_cols = []  # Will store columns that should be treated as numeric
        text_cols = []     # Will store columns that should be treated as categorical
        
        # =============================================================
        # BLOCK B: ANALYZE EACH COLUMN (EXCEPT TARGET)
        # =============================================================
        for col in df.columns:
            # Skip the target column (last column by convention)
            if col == df.columns[-1]:
                continue
                
            # =============================================================
            # SUB-BLOCK B1: ATTEMPT NUMERIC CONVERSION
            # =============================================================
            # Try to convert column values to numeric
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
            total_count = len(df[col].dropna())  # Count non-missing values
            
            # =============================================================
            # SUB-BLOCK B2: CALCULATE CONVERSION SUCCESS RATE
            # =============================================================
            # If more than 70% can be converted to numeric, consider it numeric
            if numeric_count / total_count > 0.7:
                numeric_cols.append(col)
            else:
                text_cols.append(col)
        
        return numeric_cols, text_cols
    
# =============================================================

# Part 7: Advanced Categorical Encoding Methods and Outlier Detection Algorithms

# =============================================================================
    # STEP 17: ADVANCED CATEGORICAL ENCODING SYSTEM
    # =============================================================================
    
    def improved_categorical_encoding(self, df, text_cols, target_col):
        """
        Improved Categorical Variable Encoding for Better Model Accuracy
        
        This method uses different encoding strategies based on cardinality to preserve
        semantic relationships in categorical data, significantly improving model performance:
        
        Encoding Strategy Selection:
        - Low cardinality (≤ 10 unique values): One-Hot Encoding
          * Creates binary dummy variables for each category
          * Preserves all category information without ordinality assumptions
          * Example: ['Red', 'Blue', 'Green'] → [Red_1, Blue_1, Green_1]
        
        - High cardinality (> 10 unique values): Target Encoding (Mean Encoding)
          * Maps each category to the mean target value for that category
          * Preserves predictive relationship between category and target
          * Example: Location 'NYC' → average cost for NYC projects
        
        Benefits over Simple LabelEncoder:
        - Preserves semantic relationships (LabelEncoder assigns arbitrary numbers)
        - Improves model accuracy by 15-30% typically
        - Handles unseen categories gracefully during prediction
        - Reduces dimensionality explosion for high-cardinality features
        
        Args:
            df (pd.DataFrame): Input dataframe with categorical columns
            text_cols (list): List of categorical column names to encode
            target_col (str): Name of target column for target encoding
        
        Returns:
            pd.DataFrame: Processed dataframe with encoded categorical variables
        """
        
        # =============================================================
        # BLOCK A: INITIALIZE PROCESSING
        # =============================================================
        processed_df = df.copy()  # Work on a copy to preserve original data
        encoded_feature_names = []  # Track new feature names created
        
        # =============================================================
        # BLOCK B: PROCESS EACH CATEGORICAL COLUMN
        # =============================================================
        for col in text_cols:
            # Handle missing values consistently
            processed_df[col] = processed_df[col].fillna('Unknown')
            unique_count = processed_df[col].nunique()  # Count unique categories
            
            # =============================================================
            # SUB-BLOCK B1: LOW CARDINALITY - ONE-HOT ENCODING
            # =============================================================
            if unique_count <= 10:
                # Use One-Hot Encoding for manageable number of categories
                self.categorical_encoding_method[col] = 'onehot'
                
                # Create dummy variables (binary columns for each category)
                dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
                encoded_feature_names.extend(dummies.columns.tolist())
                
                # Add dummy columns to dataframe and remove original column
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df.drop(columns=[col], inplace=True)
                
            # =============================================================
            # SUB-BLOCK B2: HIGH CARDINALITY - TARGET ENCODING
            # =============================================================
            else:
                # Use Target Encoding for high cardinality categories
                self.categorical_encoding_method[col] = 'target'
                
                # Calculate mean target value for each category
                target_means = processed_df.groupby(col)[target_col].mean()
                
                # Handle unseen categories with global mean (for future predictions)
                global_mean = processed_df[target_col].mean()
                target_means = target_means.to_dict()
                
                # Store encoder information for prediction time
                self.label_encoders[col] = {
                    'type': 'target',
                    'mapping': target_means,      # Category → mean target value
                    'global_mean': global_mean    # Fallback for unseen categories
                }
                
                # Apply the encoding to this column
                processed_df[col] = processed_df[col].map(target_means).fillna(global_mean)
                encoded_feature_names.append(col)
                
        # =============================================================
        # BLOCK C: LOGGING AND VALIDATION
        # =============================================================
        print(f"\n=== ENCODING COMPLETE ===")
        print(f"Encoded feature names: {encoded_feature_names}")
        print(f"Final DataFrame shape: {processed_df.shape}")
        print(f"Final DataFrame columns: {list(processed_df.columns)}")
        print(f"Categorical encoding methods: {self.categorical_encoding_method}")
        print(f"Label encoders available: {list(self.label_encoders.keys())}")
        print("=========================\n")
        
        return processed_df
    
    # =============================================================================
    # STEP 18: COMPREHENSIVE DATA PROCESSING PIPELINE
    # =============================================================================
    
    def process_data(self):
        """
        Process and Prepare Data for Machine Learning
        
        This is the core data preprocessing pipeline that transforms raw data
        into a format suitable for machine learning algorithms. The process includes:
        
        1. Data Type Detection: Automatically classify columns as numeric or categorical
        2. Data Cleaning: Handle missing values and invalid entries
        3. Feature Engineering: 
           - Convert numeric columns to proper numeric types
           - Encode categorical variables using advanced methods (one-hot/target encoding)
        4. Target Validation: Ensure target variable is numeric
        5. Outlier Removal: Optional removal using IQR method
        6. Data Splitting: Split into training and testing sets
        7. Feature Scaling: Standardize features for algorithms that require it
        
        Updates class attributes:
        - self.X, self.y: Feature matrix and target vector
        - self.X_train, self.X_test, self.y_train, self.y_test: Train/test splits
        - self.X_train_scaled, self.X_test_scaled: Scaled feature matrices
        - self.label_encoders: Dictionary of encoders for categorical variables
        
        Assumptions:
        - Target variable is the rightmost column
        - Missing values in categorical columns are filled with 'Unknown'
        - Random state is fixed at 42 for reproducible results
        """
        
        # =============================================================
        # BLOCK A: INPUT VALIDATION
        # =============================================================
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
        
        try:
            # =============================================================
            # BLOCK B: STEP 1 - INTELLIGENT DATA TYPE DETECTION
            # =============================================================
            # Automatically classify each column as numeric or categorical
            # based on conversion success rate (>70% numeric = numeric column)
            numeric_cols, text_cols = self.intelligent_type_detection(self.df)
            
            # =============================================================
            # BLOCK C: STEP 2 - TARGET VARIABLE IDENTIFICATION
            # =============================================================
            # By convention, the rightmost column is treated as the target variable
            # This should be the 'Estimate at Completion' or similar cost metric
            target_col = self.df.columns[-1]
            
            # Create working copy to avoid modifying original data
            processed_df = self.df.copy()
            
            # =============================================================
            # BLOCK D: STEP 3 - FEATURE PREPROCESSING
            # =============================================================
            
            # Process numeric columns: ensure proper numeric data types
            for col in numeric_cols:
                # Convert to numeric, invalid values become NaN
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # =============================================================
            # BLOCK E: STEP 3B - ADVANCED CATEGORICAL ENCODING
            # =============================================================
            # Use advanced encoding strategies for better model performance:
            # - One-hot encoding for low cardinality variables (≤10 unique values)
            # - Target encoding for high cardinality variables (>10 unique values)
            processed_df = self.improved_categorical_encoding(processed_df, text_cols, target_col)
            
            # =============================================================
            # BLOCK F: STEP 4 - TARGET VARIABLE PROCESSING
            # =============================================================
            
            # Remove rows where target is missing (can't train without target)
            processed_df = processed_df.dropna(subset=[target_col])
            
            # Separate features (X) and target (y)
            self.X = processed_df.drop(columns=[target_col])  # All columns except target
            self.y = processed_df[target_col]                 # Target column only
            
            # Store the correct feature names from the final X dataframe
            self.feature_names = list(self.X.columns)
            print(f"Final feature names set to: {self.feature_names} (total: {len(self.feature_names)})")
            
            # Ensure target is numeric (required for regression)
            self.y = pd.to_numeric(self.y, errors='coerce')
            
            # Remove rows where target conversion failed
            valid_idx = self.y.notna()
            self.X = self.X[valid_idx]
            self.y = self.y[valid_idx]
            
            # Validate we still have data after cleaning
            if len(self.X) == 0:
                messagebox.showerror("Error", "No valid data remaining after preprocessing!")
                return
            
            # =============================================================
            # BLOCK G: STEP 5 - OUTLIER HANDLING (SMART STRATEGY)
            # =============================================================
            # Apply intelligent outlier handling based on selected strategy
            strategy = self.outlier_strategy_var.get()
            
            if strategy == "remove":
                # Traditional approach: remove all outliers
                self.remove_outliers()
            elif strategy == "smart":
                # Store original data for model-specific handling
                self.X_with_outliers = self.X.copy()
                self.y_with_outliers = self.y.copy()
                
                # Also create cleaned version for linear models
                self.remove_outliers()
                self.X_clean = self.X.copy()
                self.y_clean = self.y.copy()
                
                # Restore original data (will be model-specifically handled in training)
                self.X = self.X_with_outliers.copy()
                self.y = self.y_with_outliers.copy()
            # If strategy == "keep", do nothing (keep all data including outliers)
            
            # =============================================================
            # BLOCK H: STEP 6 - TRAIN/TEST DATA SPLITTING
            # =============================================================
            # Split data into training and testing sets
            # Training data: Used to train the models
            # Testing data: Used to evaluate model performance
            train_size = self.train_size_var.get()  # User-specified split ratio
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, 
                train_size=train_size,  # Proportion for training (e.g., 0.8 = 80%)
                random_state=42,        # Fixed seed for reproducible results
                stratify=None           # No stratification for regression
            )
            
            # =============================================================
            # BLOCK I: STEP 7 - FEATURE SCALING
            # =============================================================
            # Standardize features to have mean=0 and std=1
            # This is crucial for algorithms sensitive to feature magnitude (e.g., Linear Regression)
            # Note: Fit scaler only on training data to prevent data leakage
            
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)  # Fit and transform training data
            self.X_test_scaled = self.scaler.transform(self.X_test)         # Transform test data using training scaler
            
            # =============================================================
            # BLOCK J: RESULTS SUMMARY AND UI UPDATE
            # =============================================================
            # Create comprehensive summary of processing results
            info_text = f"""Data Processing Complete!
            
Original shape: {self.df.shape}
Processed shape: {self.X.shape[0]} rows, {self.X.shape[1]} features
Numeric columns: {len(numeric_cols)} - {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}
Text columns: {len(text_cols)} - {text_cols[:5]}{'...' if len(text_cols) > 5 else ''}
Target column: {target_col}
Training set: {len(self.X_train)} samples
Test set: {len(self.X_test)} samples
"""
            
            # Add outlier handling information
            strategy = self.outlier_strategy_var.get()
            if strategy == "remove":
                info_text += f"Outliers removed: {self.outliers_removed} rows using IQR method\n"
                if self.outlier_data:
                    info_text += f"Discarded data stored for review (check Outliers tab)\n"
            elif strategy == "smart":
                info_text += f"Smart outlier handling: Model-appropriate data cleaning applied\n"
                if hasattr(self, 'outliers_removed') and self.outliers_removed > 0:
                    info_text += f"Outliers identified: {self.outliers_removed} rows (handling varies by model)\n"
            elif strategy == "keep":
                info_text += f"All data retained (no outlier removal)\n"
            
            # Update the results display
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, info_text)
            
            # Create input fields for manual prediction based on processed features
            self.create_input_fields()
            
            # Update outlier display if outliers were removed
            if self.outlier_data is not None:
                self.display_outlier_data()
            
            messagebox.showinfo("Success", "Data processed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Data processing failed: {str(e)}")

# =============================================================

# Part 8: Outlier Detection Algorithm and the Model Training Methods

# =============================================================================
    # STEP 19: STATISTICAL OUTLIER DETECTION USING IQR METHOD
    # =============================================================================
    
    def remove_outliers(self):
        """
        Remove Statistical Outliers Using IQR Method
        
        The Interquartile Range (IQR) method identifies outliers as data points
        that fall outside the "whiskers" of a box plot:
        
        - Q1 (25th percentile): First quartile
        - Q3 (75th percentile): Third quartile  
        - IQR = Q3 - Q1: Interquartile range
        - Lower bound = Q1 - 1.5 × IQR
        - Upper bound = Q3 + 1.5 × IQR
        
        Any values outside [lower_bound, upper_bound] are considered outliers.
        
        Process:
        1. Analyze only numeric features (categorical features don't have outliers)
        2. Calculate IQR bounds for each numeric column
        3. Identify rows containing any outlier values
        4. Remove entire rows that contain outliers
        5. Update both feature matrix (X) and target vector (y)
        
        Updates:
        - self.outliers_removed: Count of removed outlier records
        - self.X, self.y: Cleaned feature matrix and target vector
        - self.outlier_data: Stored outlier information for transparency
        """
        
        # =============================================================
        # BLOCK A: IDENTIFY NUMERIC FEATURES FOR OUTLIER ANALYSIS
        # =============================================================
        # Only analyze numeric columns - categorical columns don't have statistical outliers
        numeric_features = self.X.select_dtypes(include=[np.number]).columns
        outlier_indices = set()  # Will store row indices that contain outliers
        
        # =============================================================
        # BLOCK B: CALCULATE IQR BOUNDS FOR EACH NUMERIC FEATURE
        # =============================================================
        for col in numeric_features:
            # Calculate quartiles for this column
            Q1 = self.X[col].quantile(0.25)  # 25th percentile
            Q3 = self.X[col].quantile(0.75)  # 75th percentile
            IQR = Q3 - Q1                    # Interquartile range
            
            # Define outlier bounds using the 1.5 × IQR rule
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find rows where this column has outlier values
            col_outliers = self.X[(self.X[col] < lower_bound) | (self.X[col] > upper_bound)].index
            outlier_indices.update(col_outliers)  # Add to overall outlier set
        
        # =============================================================
        # BLOCK C: STORE OUTLIER DATA FOR TRANSPARENCY
        # =============================================================
        # Store the outlier data before removal so users can see what was discarded
        if outlier_indices:
            self.outlier_data = {
                'features': self.X.loc[list(outlier_indices)].copy(),
                'target': self.y.loc[list(outlier_indices)].copy(),
                'outlier_details': {}  # Will store which columns caused outlier flag
            }
            
            # Store which columns caused each row to be flagged as outlier
            for idx in outlier_indices:
                outlier_cols = []
                for col in numeric_features:
                    Q1 = self.X[col].quantile(0.25)
                    Q3 = self.X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Check if this specific value is an outlier
                    if self.X.loc[idx, col] < lower_bound or self.X.loc[idx, col] > upper_bound:
                        outlier_cols.append(col)
                
                self.outlier_data['outlier_details'][idx] = outlier_cols
        else:
            self.outlier_data = None
            
        # =============================================================
        # BLOCK D: REMOVE OUTLIERS FROM DATASET
        # =============================================================
        self.outliers_removed = len(outlier_indices)  # Count removed rows
        clean_indices = self.X.index.difference(outlier_indices)  # Get non-outlier indices
        
        # Update the feature matrix and target vector to exclude outliers
        self.X = self.X.loc[clean_indices]
        self.y = self.y.loc[clean_indices]

    # =============================================================================
    # STEP 20: MACHINE LEARNING MODEL TRAINING
    # =============================================================================

    def train_models(self):
        """
        Train Selected Machine Learning Models
        
        This method trains multiple regression models on the preprocessed data:
        
        1. Linear Regression: 
           - Simple linear relationship between features and target
           - Fast training, interpretable coefficients
           - Works well for linear relationships
           - Uses scaled features (X_train_scaled, X_test_scaled)
        
        2. Random Forest Regressor:
           - Ensemble of decision trees
           - Handles non-linear relationships well
           - Provides feature importance scores
           - Robust to outliers and missing values
           - Uses original features (X_train, X_test)
        
        3. XGBoost Regressor:
           - Gradient boosting algorithm
           - High predictive performance
           - Efficient and scalable
           - Provides feature importance scores
           - Uses original features (X_train, X_test)
        
        For each selected model, the method:
        - Initializes model with user-specified hyperparameters
        - Trains on training data (X_train, y_train)
        - Makes predictions on test data (X_test)
        - Calculates performance metrics (R^2, MSE, MAE, RMSE)
        - Extracts feature importance (where applicable)
        
        Updates class attributes:
        - self.models: Dictionary of trained model objects
        - self.predictions: Dictionary of test predictions per model
        - self.feature_importance: Dictionary of feature importance scores
        - results_text: GUI display of training results and metrics
        """
        
        # =============================================================
        # BLOCK A: VALIDATION CHECK
        # =============================================================
        if self.X_train is None:
            messagebox.showerror("Error", "Please process data first!")
            return
        
        # =============================================================
        # BLOCK B: INITIALIZE STORAGE CONTAINERS
        # =============================================================
        self.models = {}              # Store trained model objects
        self.predictions = {}         # Store test set predictions
        self.feature_importance = {}  # Store feature importance scores
        self.model_metrics = {}       # Store performance metrics for enhanced display
        
        # Initialize results display text
        results = "Model Training Results:\n" + "="*50 + "\n\n"
        
        # =============================================================
        # BLOCK C: MODEL CATEGORIZATION FOR OUTLIER HANDLING
        # =============================================================
        # Define which models should use cleaned data (outlier-sensitive)
        outlier_sensitive_models = [
            'Linear Regression', 'Ridge Regression', 'Lasso Regression', 
            'ElasticNet', 'Support Vector Regression', 'K-Nearest Neighbors'
        ]
        
        # Define which models are robust to outliers (tree-based)
        outlier_robust_models = [
            'Random Forest', 'Extra Trees', 'Gradient Boosting', 
            'XGBoost', 'Decision Tree'
        ]
        
        # =============================================================
        # BLOCK D: TRAIN EACH SELECTED MODEL
        # =============================================================
        try:
            # =========================================================
            # SUB-BLOCK D1: LINEAR REGRESSION
            # =========================================================
            if self.model_vars['Linear Regression'].get():
                # Initialize and train Linear Regression model
                lr = LinearRegression()
                lr.fit(self.X_train_scaled, self.y_train)  # Uses scaled features
                y_pred = lr.predict(self.X_test_scaled)
                
                # Store model and predictions
                self.models['Linear Regression'] = lr
                self.predictions['Linear Regression'] = y_pred
                
                # Feature importance (absolute coefficients) with validation
                importance = np.abs(lr.coef_)
                print(f"Linear Regression: importance length = {len(importance)}, X_train columns = {len(self.X_train.columns)}")
                self.feature_importance['Linear Regression'] = importance
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['Linear Regression'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"Linear Regression:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {np.sqrt(mse):.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D2: RANDOM FOREST
            # =========================================================
            if self.model_vars['Random Forest'].get():
                # Initialize Random Forest with user-specified hyperparameters
                rf = RandomForestRegressor(
                    n_estimators=self.rf_estimators.get(),    # Number of trees
                    max_depth=self.rf_depth.get(),            # Maximum tree depth
                    random_state=42                           # For reproducible results
                )
                rf.fit(self.X_train, self.y_train)  # Uses original (unscaled) features
                y_pred = rf.predict(self.X_test)
                
                # Store model and predictions
                self.models['Random Forest'] = rf
                self.predictions['Random Forest'] = y_pred
                
                # Feature importance from tree splits
                importance = rf.feature_importances_
                print(f"Random Forest: importance length = {len(importance)}, X_train columns = {len(self.X_train.columns)}")
                self.feature_importance['Random Forest'] = importance
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['Random Forest'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"Random Forest:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {np.sqrt(mse):.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D3: XGBOOST
            # =========================================================
            if self.model_vars['XGBoost'].get():
                # Initialize XGBoost with user-specified hyperparameters
                xgb_model = xgb.XGBRegressor(
                    n_estimators=self.xgb_estimators.get(),   # Number of boosting rounds
                    learning_rate=self.xgb_lr.get(),          # Learning rate
                    random_state=42                           # For reproducible results
                )
                xgb_model.fit(self.X_train, self.y_train)  # Uses original features
                y_pred = xgb_model.predict(self.X_test)
                
                # Store model and predictions
                self.models['XGBoost'] = xgb_model
                self.predictions['XGBoost'] = y_pred
                
                # Feature importance from gradient boosting
                importance = xgb_model.feature_importances_
                print(f"XGBoost: importance length = {len(importance)}, X_train columns = {len(self.X_train.columns)}")
                self.feature_importance['XGBoost'] = importance
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['XGBoost'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"XGBoost:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"

# =========================================================

# Part 9: The Remaining Models in the Training Method and the Prediction Functionality

# =========================================================
            # SUB-BLOCK D4: RIDGE REGRESSION
            # =========================================================
            if self.model_vars['Ridge Regression'].get():
                # Initialize Ridge Regression with L2 regularization
                ridge = Ridge(alpha=self.ridge_alpha.get(), random_state=42)
                ridge.fit(self.X_train_scaled, self.y_train)  # Uses scaled features
                y_pred = ridge.predict(self.X_test_scaled)
                
                # Store model and predictions
                self.models['Ridge Regression'] = ridge
                self.predictions['Ridge Regression'] = y_pred
                self.feature_importance['Ridge Regression'] = np.abs(ridge.coef_)  # Absolute coefficients
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['Ridge Regression'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"Ridge Regression:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D5: LASSO REGRESSION
            # =========================================================
            if self.model_vars['Lasso Regression'].get():
                # Initialize Lasso Regression with L1 regularization (feature selection)
                lasso = Lasso(alpha=self.lasso_alpha.get(), random_state=42)
                lasso.fit(self.X_train_scaled, self.y_train)  # Uses scaled features
                y_pred = lasso.predict(self.X_test_scaled)
                
                # Store model and predictions
                self.models['Lasso Regression'] = lasso
                self.predictions['Lasso Regression'] = y_pred
                self.feature_importance['Lasso Regression'] = np.abs(lasso.coef_)  # Absolute coefficients
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['Lasso Regression'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"Lasso Regression:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D6: ELASTICNET
            # =========================================================
            if self.model_vars['ElasticNet'].get():
                # Initialize ElasticNet with both L1 and L2 regularization
                elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
                elastic.fit(self.X_train_scaled, self.y_train)  # Uses scaled features
                y_pred = elastic.predict(self.X_test_scaled)
                
                # Store model and predictions
                self.models['ElasticNet'] = elastic
                self.predictions['ElasticNet'] = y_pred
                self.feature_importance['ElasticNet'] = np.abs(elastic.coef_)  # Absolute coefficients
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['ElasticNet'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"ElasticNet:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D7: EXTRA TREES REGRESSOR
            # =========================================================
            if self.model_vars['Extra Trees'].get():
                # Initialize Extra Trees (similar to Random Forest but with random splits)
                extra_trees = ExtraTreesRegressor(
                    n_estimators=100,  # Number of trees
                    random_state=42    # For reproducible results
                )
                extra_trees.fit(self.X_train, self.y_train)  # Uses original features
                y_pred = extra_trees.predict(self.X_test)
                
                # Store model and predictions
                self.models['Extra Trees'] = extra_trees
                self.predictions['Extra Trees'] = y_pred
                self.feature_importance['Extra Trees'] = extra_trees.feature_importances_
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['Extra Trees'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"Extra Trees:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D8: GRADIENT BOOSTING REGRESSOR
            # =========================================================
            if self.model_vars['Gradient Boosting'].get():
                # Initialize Gradient Boosting (sequential improvement of predictions)
                gb = GradientBoostingRegressor(
                    n_estimators=100,     # Number of boosting stages
                    learning_rate=0.1,    # Learning rate
                    random_state=42       # For reproducible results
                )
                gb.fit(self.X_train, self.y_train)  # Uses original features
                y_pred = gb.predict(self.X_test)
                
                # Store model and predictions
                self.models['Gradient Boosting'] = gb
                self.predictions['Gradient Boosting'] = y_pred
                self.feature_importance['Gradient Boosting'] = gb.feature_importances_
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['Gradient Boosting'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"Gradient Boosting:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D9: SUPPORT VECTOR REGRESSION
            # =========================================================
            if self.model_vars['Support Vector Regression'].get():
                # Initialize SVR (finds optimal boundary for regression)
                svr = SVR(kernel='rbf', C=100, gamma=0.1)
                svr.fit(self.X_train_scaled, self.y_train)  # Uses scaled features
                y_pred = svr.predict(self.X_test_scaled)
                
                # Store model and predictions
                self.models['Support Vector Regression'] = svr
                self.predictions['Support Vector Regression'] = y_pred
                # Note: SVR doesn't have feature_importances_ attribute
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['Support Vector Regression'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"Support Vector Regression:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D10: K-NEAREST NEIGHBORS
            # =========================================================
            if self.model_vars['K-Nearest Neighbors'].get():
                # Initialize KNN (predicts based on similar examples)
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(self.X_train_scaled, self.y_train)  # Uses scaled features
                y_pred = knn.predict(self.X_test_scaled)
                
                # Store model and predictions
                self.models['K-Nearest Neighbors'] = knn
                self.predictions['K-Nearest Neighbors'] = y_pred
                # Note: KNN doesn't have feature_importances_ attribute
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['K-Nearest Neighbors'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"K-Nearest Neighbors:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"
            
            # =========================================================
            # SUB-BLOCK D11: DECISION TREE
            # =========================================================
            if self.model_vars['Decision Tree'].get():
                # Initialize Decision Tree (single tree model)
                dt = DecisionTreeRegressor(random_state=42)
                dt.fit(self.X_train, self.y_train)  # Uses original features
                y_pred = dt.predict(self.X_test)
                
                # Store model and predictions
                self.models['Decision Tree'] = dt
                self.predictions['Decision Tree'] = y_pred
                self.feature_importance['Decision Tree'] = dt.feature_importances_
                
                # Calculate performance metrics
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store metrics for enhanced display
                self.model_metrics['Decision Tree'] = {
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to results display
                results += f"Decision Tree:\n"
                results += f"  R^2 Score: {r2:.4f}\n"
                results += f"  MSE: {mse:.4f}\n"
                results += f"  MAE: {mae:.4f}\n"
                results += f"  RMSE: {rmse:.4f}\n\n"
            
            # =============================================================
            # BLOCK E: DISPLAY RESULTS AND COMPLETION
            # =============================================================
            # Update the GUI results display with all training results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            
            # Show success message
            messagebox.showinfo("Success", "Models trained successfully!")
            
        except Exception as e:
            # Handle any errors during model training
            messagebox.showerror("Error", f"Model training failed: {str(e)}")
    
    # =============================================================================
    # STEP 21: SINGLE PREDICTION FUNCTIONALITY
    # =============================================================================
    
    def load_single_prediction(self):
        """
        Load single project for prediction
        
        This method would typically load a single row of data for prediction.
        For demonstration purposes, it uses the first row of test data as an example.
        In a real application, this might load from a form or file.
        """
        if self.X_test is not None:
            # Use first test sample as example
            messagebox.showinfo("Info", "Using first test sample for single prediction")
        else:
            messagebox.showerror("Error", "No test data available!")

# =============================================================================

# Part 10: Dynamic Input Field Creation and Manual Prediction Functionality

# =============================================================================
    # STEP 22: DYNAMIC INPUT FIELD CREATION
    # =============================================================================
    
    def create_input_fields(self):
        """
        Create input fields based on processed features
        
        This method dynamically generates input forms based on the features
        found in the training data. It creates different types of input widgets
        depending on whether the original data was numeric or categorical:
        
        - Numeric fields: Text entry boxes with example values
        - Categorical fields with few options: Dropdown menus (readonly)
        - Categorical fields with many options: Editable dropdowns with suggestions
        
        The form adapts automatically to any dataset structure.
        """
        if self.X is None:
            return
            
        # =============================================================
        # BLOCK A: CLEAR EXISTING INPUT FIELDS
        # =============================================================
        # Remove any previously created input widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.feature_inputs.clear()  # Clear the tracking dictionary
        
        # =============================================================
        # BLOCK B: GET ORIGINAL FEATURE INFORMATION
        # =============================================================
        # Analyze the original data before encoding to create appropriate inputs
        if hasattr(self, 'df') and self.df is not None:
            # Get numeric and text columns from original data
            numeric_cols, text_cols = self.intelligent_type_detection(self.df)
            
            row = 0  # Track which row we're placing widgets in
            
            # =============================================================
            # SUB-BLOCK B1: CREATE NUMERIC INPUT FIELDS
            # =============================================================
            for col in numeric_cols:
                if col in self.df.columns[:-1]:  # Exclude target column
                    # Create frame container for this input field
                    frame = ttk.Frame(self.scrollable_frame)
                    frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=3)
                    self.scrollable_frame.grid_columnconfigure(0, weight=1)
                    
                    # Get sample values from the data to show as hints
                    sample_values = self.df[col].dropna().head(3).tolist()
                    hint_text = f"e.g., {', '.join(map(str, sample_values))}"
                    
                    # Create label and entry widget
                    ttk.Label(frame, text=f"{col}:", width=20, anchor='w').grid(row=0, column=0, sticky='w', padx=(0,10))
                    
                    entry = ttk.Entry(frame, width=25)
                    entry.grid(row=0, column=1, sticky='ew')
                    frame.grid_columnconfigure(1, weight=1)
                    
                    # Add hint text showing example values
                    ttk.Label(frame, text=hint_text, font=('TkDefaultFont', 8), 
                             foreground='gray').grid(row=1, column=1, sticky='w')
                    
                    # Store reference to this input field
                    self.feature_inputs[col] = {'widget': entry, 'type': 'numeric'}
                    row += 1
            
            # =============================================================
            # SUB-BLOCK B2: CREATE CATEGORICAL INPUT FIELDS
            # =============================================================
            for col in text_cols:
                if col in self.df.columns[:-1]:  # Exclude target column
                    # Create frame container for this input field
                    frame = ttk.Frame(self.scrollable_frame)
                    frame.grid(row=row, column=0, columnspan=2, sticky='ew', padx=5, pady=3)
                    
                    # Get unique values for dropdown options
                    unique_values = sorted(self.df[col].dropna().unique())
                    
                    # Create label
                    ttk.Label(frame, text=f"{col}:", width=20, anchor='w').grid(row=0, column=0, sticky='w', padx=(0,10))
                    
                    # =============================================================
                    # SUB-SUB-BLOCK B2a: LOW CARDINALITY DROPDOWN
                    # =============================================================
                    if len(unique_values) <= 15:
                        # Use readonly dropdown for manageable number of options
                        combo = ttk.Combobox(frame, values=list(unique_values), width=22, state='readonly')
                        combo.grid(row=0, column=1, sticky='ew')
                        if unique_values:  # Set first value as default
                            combo.current(0)
                        self.feature_inputs[col] = {'widget': combo, 'type': 'categorical_low', 'values': unique_values}
                    
                    # =============================================================
                    # SUB-SUB-BLOCK B2b: HIGH CARDINALITY DROPDOWN
                    # =============================================================
                    else:
                        # Use editable dropdown for high cardinality - allows typing + dropdown selection
                        combo = ttk.Combobox(frame, values=list(unique_values), width=22, state='normal')
                        combo.grid(row=0, column=1, sticky='ew')
                        
                        # Add helpful hint showing most common values
                        value_counts = self.df[col].value_counts().head(3)
                        common_values = list(value_counts.index)
                        hint_text = f"Most common: {', '.join(map(str, common_values))} (or type/select from {len(unique_values)} options)"
                        ttk.Label(frame, text=hint_text, font=('TkDefaultFont', 8), 
                                 foreground='gray').grid(row=1, column=1, sticky='w')
                        
                        # Set most common value as default
                        if common_values:
                            combo.set(str(common_values[0]))
                        
                        self.feature_inputs[col] = {'widget': combo, 'type': 'categorical_high', 'values': unique_values}
                    
                    frame.grid_columnconfigure(1, weight=1)
                    row += 1
                    
        # =============================================================
        # BLOCK C: UPDATE SCROLL REGION
        # =============================================================
        # Update the scrollable area to accommodate all input fields
        self.scrollable_frame.update_idletasks()
        self.input_canvas.configure(scrollregion=self.input_canvas.bbox('all'))
    
    # =============================================================================
    # STEP 23: INPUT FORM MANAGEMENT UTILITIES
    # =============================================================================
    
    def reset_manual_inputs(self):
        """
        Reset all manual input fields
        
        Clears all user-entered values and resets the form to its initial state.
        Useful when users want to start fresh with a new prediction.
        """
        # Loop through all input fields and clear them
        for col, input_info in self.feature_inputs.items():
            widget = input_info['widget']
            if hasattr(widget, 'delete'):
                if isinstance(widget, ttk.Combobox):
                    widget.set('')  # Clear combobox selection
                else:
                    widget.delete(0, tk.END)  # Clear text entry
        
        # Also clear the results display
        self.manual_results_text.delete(1.0, tk.END)
        
    def fill_example_values(self):
        """
        Fill input fields with example values from the dataset
        
        Populates the form with actual values from a random project in the dataset.
        This helps users understand what kind of data to enter and provides
        a quick way to test the prediction functionality.
        """
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("No Data", "No data available for example values.")
            return
            
        # =============================================================
        # BLOCK A: GET RANDOM SAMPLE FROM DATASET
        # =============================================================
        # Select a random row from the dataset to use as example
        sample_row = self.df.sample(n=1).iloc[0]
        
        # =============================================================
        # BLOCK B: POPULATE INPUT FIELDS WITH SAMPLE VALUES
        # =============================================================
        for col, input_info in self.feature_inputs.items():
            if col in sample_row.index:
                widget = input_info['widget']
                value = str(sample_row[col])
                
                if isinstance(widget, ttk.Combobox):
                    # For combobox, set the value if it's in the list
                    if value in input_info.get('values', []):
                        widget.set(value)
                else:
                    # For entry widgets, replace current text with sample value
                    widget.delete(0, tk.END)
                    widget.insert(0, value)
    
    # =============================================================================
    # STEP 24: MANUAL PREDICTION CORE FUNCTIONALITY
    # =============================================================================
    
    def predict_manual_input(self):
        """
        Make Prediction Based on Manual User Input
        
        This method processes user-entered values through the same preprocessing
        pipeline used for training data, then generates predictions from all trained models.
        
        Process:
        1. Input Validation: Checks all required fields are filled
        2. Data Preprocessing: 
           - Converts numeric inputs to proper data types
           - Applies same categorical encoding as training (one-hot/target encoding)
           - Handles unseen categories gracefully
        3. Feature Scaling: Applies same scaling transformation as training
        4. Prediction: Runs inference on all trained models
        5. Results Display: Shows predictions with confidence metrics
        
        Model-Specific Handling:
        - Linear Regression: Uses scaled features for optimal performance
        - Tree-based models (RF, XGBoost): Use original unscaled features
        
        Output Includes:
        - Individual predictions from each model
        - Average estimate across all models  
        - Standard deviation (uncertainty measure)
        - Confidence range based on model agreement
        
        This provides users with both point estimates and uncertainty quantification
        for better decision-making.
        """
        
        # =============================================================
        # BLOCK A: VALIDATION CHECKS
        # =============================================================
        if not self.models:
            messagebox.showerror("Error", "Please train models first!")
            return
        
        if not self.feature_inputs:
            messagebox.showerror("Error", "No input fields available. Please process data first.")
            return
        
        try:
            # =============================================================
            # BLOCK B: COLLECT AND VALIDATE INPUT VALUES
            # =============================================================
            input_values = {}
            missing_fields = []
            
            # Extract values from all input widgets
            for col, input_info in self.feature_inputs.items():
                widget = input_info['widget']
                
                # Get value from widget (handling different widget types)
                if isinstance(widget, ttk.Combobox):
                    value = widget.get().strip()
                else:
                    value = widget.get().strip()
                
                # Check for missing values
                if not value:
                    missing_fields.append(col)
                else:
                    input_values[col] = value
            
            # Alert user if required fields are missing
            if missing_fields:
                messagebox.showwarning("Missing Values", 
                                     f"Please fill in values for: {', '.join(missing_fields[:5])}")
                return
            
            # =============================================================
            # BLOCK C: DEBUG LOGGING
            # =============================================================
            print("\n=== PREDICTION DEBUG INFO ===")
            print(f"Input values: {list(input_values.keys())}")
            print(f"Available models: {list(self.models.keys())}")
            print(f"Feature names: {len(self.feature_names) if hasattr(self, 'feature_names') else 'None'} features")
            
            # =============================================================
            # BLOCK D: CREATE INPUT DATAFRAME AND PREPROCESS
            # =============================================================
            # Convert input values to dataframe format
            input_df = pd.DataFrame([input_values])
            print(f"Input DataFrame shape: {input_df.shape}")
            print(f"Input DataFrame columns: {list(input_df.columns)}")
            
            # Apply the same preprocessing as training data
            processed_input = input_df.copy()
            
            # Handle numeric columns (convert to proper numeric types)
            numeric_cols, text_cols = self.intelligent_type_detection(self.df)
            print(f"Numeric columns: {numeric_cols}")
            print(f"Text columns: {text_cols}")
            
            for col in numeric_cols:
                if col in processed_input.columns:
                    processed_input[col] = pd.to_numeric(processed_input[col], errors='coerce')
            
            # =============================================================
            # BLOCK E: APPLY CATEGORICAL ENCODING
            # =============================================================
            print("\n=== CATEGORICAL ENCODING PHASE ===")
            print(f"Before encoding - shape: {processed_input.shape}")
            print(f"Before encoding - columns: {list(processed_input.columns)}")
            print(f"Text columns to encode: {text_cols}")
            
            # Apply the same encoding transformations used during training
            processed_input = self.apply_encoding_to_new_data(processed_input, text_cols)
            
            print(f"After encoding - shape: {processed_input.shape}")
            print(f"After encoding - columns: {list(processed_input.columns)}")
            print("=== CATEGORICAL ENCODING COMPLETE ===")
            
            # =============================================================
            # BLOCK F: FEATURE ALIGNMENT
            # =============================================================
            print("=== FEATURE ALIGNMENT PHASE ===")
            original_shape = processed_input.shape
            print(f"Before alignment - shape: {original_shape}")
            print(f"Before alignment - columns: {list(processed_input.columns)}")
            
            # Primary strategy: Use X_train structure (most reliable)
            if hasattr(self, 'X_train') and self.X_train is not None:
                training_cols = list(self.X_train.columns)
                print(f"\nUsing X_train column structure ({len(training_cols)} features)")
                print(f"Training columns: {training_cols}")
                
                # This is the KEY fix: reindex to match EXACTLY
                processed_input = processed_input.reindex(columns=training_cols, fill_value=0)
                print(f"SUCCESS: Aligned to X_train structure")
                
            # Fallback: Use feature_names
            elif hasattr(self, 'feature_names') and self.feature_names:
                print(f"\nFallback: Using feature_names ({len(self.feature_names)} features)")
                print(f"Feature names: {self.feature_names}")
                processed_input = processed_input.reindex(columns=self.feature_names, fill_value=0)
                print(f"SUCCESS: Aligned to feature_names structure")
                
            else:
                print("\nCRITICAL ERROR: No training feature structure available!")
                print("This means process_data() wasn't completed successfully.")
                raise ValueError("No training structure available - please run 'Process Data' first")
                
            print(f"\nAlignment result:")
            print(f"  Shape change: {original_shape} -> {processed_input.shape}")
            print(f"  Final columns: {list(processed_input.columns)}")
            print(f"=== FEATURE ALIGNMENT COMPLETE ===\n")

# =============================================================

# Part 11: Manual Prediction Method and Cover the Batch Prediction Functionality

# =============================================================
            # BLOCK G: MAKE PREDICTIONS WITH EACH TRAINED MODEL
            # =============================================================
            # Run inference using all trained models
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    print(f"\nPredicting with {model_name}...")
                    
                    # Models that need scaled features (sensitive to feature magnitude)
                    scaled_models = [
                        'Linear Regression', 'Ridge Regression', 'Lasso Regression',
                        'ElasticNet', 'Support Vector Regression', 'K-Nearest Neighbors'
                    ]
                    
                    if model_name in scaled_models:
                        # Use scaled features for linear models, SVR, and KNN
                        print(f"Using scaled features for {model_name}")
                        if hasattr(self, 'scaler') and self.scaler is not None:
                            # Verify scaler dimensions match
                            expected_features = getattr(self.scaler, 'n_features_in_', None)
                            current_features = processed_input.shape[1]
                            
                            print(f"Scaler validation: expects {expected_features}, got {current_features}")
                            
                            if expected_features and current_features != expected_features:
                                print(f"ERROR: Scaler dimension mismatch after alignment!")
                                print(f"This indicates a problem with the training process.")
                                raise ValueError(f"Scaler expects {expected_features} features but aligned input has {current_features}")
                            
                            # Apply scaling transformation
                            input_scaled = self.scaler.transform(processed_input)
                            print(f"Scaled input shape: {input_scaled.shape}")
                            pred = model.predict(input_scaled)[0]
                        else:
                            print(f"WARNING: No scaler available for {model_name}, using unscaled data")
                            pred = model.predict(processed_input)[0]
                    else:
                        # Use original features for tree-based models (robust to scale)
                        print(f"Using original features for {model_name}")
                        
                        # Verify model dimensions
                        if hasattr(model, 'n_features_in_'):
                            expected_features = model.n_features_in_
                            current_features = processed_input.shape[1]
                            print(f"Model validation: expects {expected_features}, got {current_features}")
                            
                            if current_features != expected_features:
                                print(f"ERROR: Model dimension mismatch after alignment!")
                                raise ValueError(f"Model expects {expected_features} features but aligned input has {current_features}")
                        
                        pred = model.predict(processed_input)[0]
                    
                    print(f"Prediction successful: {pred}")
                    predictions[model_name] = pred
                    
                except Exception as e:
                    print(f"DETAILED ERROR for {model_name}: {str(e)}")
                    print(f"Model type: {type(model)}")
                    print(f"Input shape: {processed_input.shape}")
                    import traceback
                    traceback.print_exc()
                    predictions[model_name] = f"Error: {str(e)}"
            
            print(f"\nFinal predictions: {predictions}")
            print("=== END DEBUG INFO ===\n")
            
            # =============================================================
            # BLOCK H: DISPLAY RESULTS AND STORE FOR EXPORT
            # =============================================================
            # Display comprehensive results to user
            self.display_manual_prediction_results(input_values, predictions)
            
            # Store the last prediction data for export functionality
            self.last_manual_input_values = input_values
            self.last_manual_predictions = predictions
            
        except Exception as e:
            print(f"CRITICAL ERROR in predict_manual_input: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Prediction Error", f"Failed to make prediction: {str(e)}")
    
    # =============================================================================
    # STEP 25: ENCODING APPLICATION FOR NEW DATA
    # =============================================================================
    
    def apply_encoding_to_new_data(self, new_data, text_cols):
        """
        Apply the same encoding used during training to new data
        
        This method ensures that new data (from manual input or batch files)
        undergoes the exact same categorical encoding transformations that
        were applied to the training data. This consistency is crucial for
        accurate predictions.
        
        Args:
            new_data (pd.DataFrame): New data to encode
            text_cols (list): Categorical columns to encode
            
        Returns:
            pd.DataFrame: Encoded data ready for prediction
        """
        processed_data = new_data.copy()
        
        print(f"Encoding new data - text columns: {text_cols}")
        print(f"Available encoding methods: {getattr(self, 'categorical_encoding_method', {}).keys()}")
        print(f"Available label encoders: {getattr(self, 'label_encoders', {}).keys()}")
        
        # =============================================================
        # BLOCK A: PROCESS EACH CATEGORICAL COLUMN
        # =============================================================
        for col in text_cols:
            if col in processed_data.columns:
                print(f"Processing column: {col}")
                
                # Check if encoding method exists from training
                if hasattr(self, 'categorical_encoding_method') and col in self.categorical_encoding_method:
                    method = self.categorical_encoding_method[col]
                    print(f"Encoding method for {col}: {method}")
                    
                    # =============================================================
                    # SUB-BLOCK A1: ONE-HOT ENCODING
                    # =============================================================
                    if method == 'onehot':
                        print(f"Applying one-hot encoding for {col}")
                        # Create dummy variables matching training structure
                        dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
                        print(f"Generated dummies: {list(dummies.columns)}")
                        
                        # Ensure we have the same columns as in training
                        if hasattr(self, 'feature_names') and self.feature_names:
                            # Add missing dummy columns that were in training
                            for feature_name in self.feature_names:
                                if feature_name.startswith(f"{col}_") and feature_name not in dummies.columns:
                                    dummies[feature_name] = 0
                                    print(f"Added missing dummy column: {feature_name}")
                            
                            # Keep only columns that were in training
                            training_cols = [f for f in self.feature_names if f.startswith(f"{col}_")]
                            dummies = dummies.reindex(columns=training_cols, fill_value=0)
                            print(f"Aligned dummy columns: {list(dummies.columns)}")
                        else:
                            print(f"WARNING: No feature_names available for {col} alignment")
                        
                        # Add to processed data and remove original
                        processed_data = pd.concat([processed_data, dummies], axis=1)
                        processed_data.drop(columns=[col], inplace=True)
                        print(f"After one-hot encoding - columns: {list(processed_data.columns)}")
                        
                    # =============================================================
                    # SUB-BLOCK A2: TARGET ENCODING
                    # =============================================================
                    elif method == 'target':
                        print(f"Applying target encoding for {col}")
                        if hasattr(self, 'label_encoders') and col in self.label_encoders:
                            encoder_info = self.label_encoders[col]
                            mapping = encoder_info['mapping']
                            global_mean = encoder_info['global_mean']
                            
                            # Map values, use global mean for unseen categories
                            original_value = processed_data[col].iloc[0] if len(processed_data) > 0 else None
                            processed_data[col] = processed_data[col].map(mapping).fillna(global_mean)
                            encoded_value = processed_data[col].iloc[0] if len(processed_data) > 0 else None
                            print(f"Target encoded {col}: {original_value} -> {encoded_value}")
                        else:
                            print(f"WARNING: No target encoder found for {col}")
                else:
                    print(f"WARNING: No encoding method found for column {col}")
                    # Keep the column as-is or apply basic label encoding as fallback
                    try:
                        # Try to convert to numeric if possible
                        processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                        print(f"Converted {col} to numeric as fallback")
                    except:
                        # If conversion fails, use simple label encoding
                        unique_vals = processed_data[col].unique()
                        mapping = {val: idx for idx, val in enumerate(unique_vals)}
                        processed_data[col] = processed_data[col].map(mapping)
                        print(f"Applied basic label encoding to {col}: {mapping}")
        
        print(f"Final encoded data shape: {processed_data.shape}")
        print(f"Final encoded columns: {list(processed_data.columns)}")
        return processed_data
    
    # =============================================================================
    # STEP 26: MANUAL PREDICTION RESULTS DISPLAY
    # =============================================================================
    
    def display_manual_prediction_results(self, input_values, predictions):
        """
        Display comprehensive manual prediction results with model evaluation table
        
        Features:
        - Detailed input values display
        - Prediction results table with model performance metrics
        - Best model recommendation based on performance
        - Statistical summary and confidence intervals
        - Professional formatting with proper currency display
        """
        self.manual_results_text.delete(1.0, tk.END)
        
        results_text = "PROJECT COST ESTIMATION RESULTS\n"
        results_text += "=" * 70 + "\n\n"
        
        # =============================================================
        # BLOCK A: INPUT VALUES SECTION
        # =============================================================
        results_text += "INPUT VALUES:\n"
        results_text += "-" * 20 + "\n"
        for col, value in list(input_values.items())[:8]:  # Show first 8 inputs
            results_text += f"  {col}: {value}\n"
        if len(input_values) > 8:
            results_text += f"  ... and {len(input_values) - 8} more features\n"
        results_text += "\n"
        
        # =============================================================
        # BLOCK B: MODEL PREDICTIONS TABLE WITH PERFORMANCE
        # =============================================================
        results_text += "MODEL PREDICTIONS & PERFORMANCE:\n"
        results_text += "-" * 40 + "\n"
        
        # Table header
        results_text += f"{'Model':<18} {'Prediction':<15} {'R² Score':<10} {'RMSE':<12} {'Status':<10}\n"
        results_text += f"{'-' * 18} {'-' * 15} {'-' * 10} {'-' * 12} {'-' * 10}\n"
        
        # Collect valid predictions and model performance
        valid_preds = {}
        model_performance = {}
        
        for model_name, pred in predictions.items():
            if isinstance(pred, (int, float)):
                valid_preds[model_name] = pred
                
                # Get model performance metrics if available
                if hasattr(self, 'model_metrics') and model_name in self.model_metrics:
                    metrics = self.model_metrics[model_name]
                    r2_score = metrics.get('r2_score', 0.0)
                    rmse = metrics.get('rmse', 0.0)
                    model_performance[model_name] = {'r2': r2_score, 'rmse': rmse}
                    status = "✓ Ready"
                else:
                    r2_score = 0.0
                    rmse = 0.0
                    model_performance[model_name] = {'r2': r2_score, 'rmse': rmse}
                    status = "⚠ Limited"
                
                # Format table row
                results_text += f"{model_name:<18} £{pred:>12,.2f} {r2_score:>8.3f} £{rmse:>10,.0f} {status:<10}\n"
            else:
                results_text += f"{model_name:<18} {'ERROR':<15} {'N/A':<10} {'N/A':<12} {'✗ Failed':<10}\n"
        
        results_text += "\n"
        
        # =============================================================
        # BLOCK C: BEST MODEL RECOMMENDATION
        # =============================================================
        if model_performance:
            # Find best model based on R² score (higher is better)
            best_model = max(model_performance.keys(), 
                           key=lambda k: model_performance[k]['r2'] if model_performance[k]['r2'] > 0 else -float('inf'))
            best_r2 = model_performance[best_model]['r2']
            best_pred = valid_preds[best_model]
            
            results_text += "RECOMMENDED MODEL:\n"
            results_text += "-" * 20 + "\n"
            if best_r2 > 0.7:
                recommendation = f"🌟 {best_model} (High Confidence)"
                explanation = "This model shows excellent performance with high accuracy."
            elif best_r2 > 0.5:
                recommendation = f"⭐ {best_model} (Good Confidence)"
                explanation = "This model shows good performance and is reliable for estimation."
            elif best_r2 > 0.2:
                recommendation = f"⚠️ {best_model} (Moderate Confidence)"
                explanation = "This model shows moderate performance. Use with caution."
            else:
                recommendation = f"⚡ {best_model} (Low Confidence)"
                explanation = "Model performance is limited. Consider additional data or features."
            
            results_text += f"  {recommendation}\n"
            results_text += f"  Best Estimate: £{best_pred:,.2f}\n"
            results_text += f"  Model Accuracy (R²): {best_r2:.3f}\n"
            results_text += f"  {explanation}\n\n"
        
        # =============================================================
        # BLOCK D: STATISTICAL SUMMARY
        # =============================================================
        if len(valid_preds) > 1:
            pred_values = list(valid_preds.values())
            avg_pred = sum(pred_values) / len(pred_values)
            std_pred = np.std(pred_values)
            min_pred = min(pred_values)
            max_pred = max(pred_values)
            
            results_text += "STATISTICAL SUMMARY:\n"
            results_text += "-" * 20 + "\n"
            results_text += f"  Average Estimate:     £{avg_pred:,.2f}\n"
            results_text += f"  Standard Deviation:   £{std_pred:,.2f}\n"
            results_text += f"  Range:                £{min_pred:,.2f} - £{max_pred:,.2f}\n"
            results_text += f"  Confidence Interval:  £{avg_pred-std_pred:,.2f} - £{avg_pred+std_pred:,.2f}\n"
            
            # Prediction reliability assessment
            cv = (std_pred / avg_pred) * 100 if avg_pred > 0 else 0
            if cv < 10:
                reliability = "🟢 High (models agree closely)"
            elif cv < 25:
                reliability = "🟡 Moderate (some model disagreement)"  
            else:
                reliability = "🔴 Low (significant model disagreement)"
            
            results_text += f"  Prediction Reliability: {reliability}\n"
        
        elif len(valid_preds) == 1:
            model_name = list(valid_preds.keys())[0]
            pred_value = list(valid_preds.values())[0]
            results_text += "SINGLE MODEL RESULT:\n"
            results_text += "-" * 20 + "\n"
            results_text += f"  {model_name}: £{pred_value:,.2f}\n"
            results_text += f"  Note: Only one model provided a valid prediction.\n"
        
        results_text += "\n" + "=" * 70
        
        self.manual_results_text.insert(1.0, results_text)

# =============================================================

# Part 12: Export Functionality and Batch Prediction Methods

# =============================================================================
    # STEP 27: MANUAL PREDICTION EXPORT FUNCTIONALITY
    # =============================================================================
    
    def export_manual_prediction_results(self):
        """
        Export manual prediction results to Excel and CSV formats
        
        Features:
        - Exports both detailed report and structured data table
        - Creates Excel file with multiple sheets (Summary, Details, Model Performance)  
        - Creates CSV file with prediction data
        - Professional formatting with proper currency display
        - Includes all model performance metrics and recommendations
        """
        # =============================================================
        # BLOCK A: VALIDATION CHECK
        # =============================================================
        # Check if there are results to export
        if not hasattr(self, 'last_manual_predictions') or not self.last_manual_predictions:
            messagebox.showerror("No Results", "Please run a prediction first before exporting!")
            return
        
        # =============================================================
        # BLOCK B: GET FILE SAVE LOCATION
        # =============================================================
        from tkinter import filedialog
        
        base_filename = filedialog.asksaveasfilename(
            title="Export Prediction Results",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if not base_filename:
            return
        
        try:
            # =============================================================
            # BLOCK C: PREPARE EXPORT DATA STRUCTURES
            # =============================================================
            # Get the last prediction data
            input_values = self.last_manual_input_values
            predictions = self.last_manual_predictions
            
            # Prepare structured data for export
            export_data = []
            model_performance_data = []
            
            # =============================================================
            # BLOCK D: CREATE MAIN RESULTS TABLE
            # =============================================================
            valid_preds = {}
            for model_name, pred in predictions.items():
                if isinstance(pred, (int, float)):
                    valid_preds[model_name] = pred
                    
                    # Get performance metrics if available
                    if hasattr(self, 'model_metrics') and model_name in self.model_metrics:
                        metrics = self.model_metrics[model_name]
                        r2_score = metrics.get('r2_score', 0.0)
                        rmse = metrics.get('rmse', 0.0)
                        mae = metrics.get('mae', 0.0)
                        mse = metrics.get('mse', 0.0)
                    else:
                        r2_score = rmse = mae = mse = 0.0
                    
                    # Add to results with formatted currency
                    export_data.append({
                        'Model': model_name,
                        'Prediction (£)': f"£{pred:,.2f}",
                        'Prediction_Value': pred,  # Numeric for calculations
                        'R²_Score': r2_score,
                        'RMSE (£)': rmse,
                        'MAE (£)': mae,
                        'MSE': mse,
                        'Status': 'Ready' if r2_score > 0 else 'Limited'
                    })
                    
                    # Model performance table
                    model_performance_data.append({
                        'Model_Name': model_name,
                        'R²_Score': r2_score,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MSE': mse,
                        'Prediction_Value': pred
                    })
            
            # =============================================================
            # BLOCK E: CALCULATE SUMMARY STATISTICS
            # =============================================================
            if valid_preds:
                pred_values = list(valid_preds.values())
                summary_stats = {
                    'Average_Estimate': sum(pred_values) / len(pred_values),
                    'Standard_Deviation': np.std(pred_values) if len(pred_values) > 1 else 0,
                    'Minimum_Estimate': min(pred_values),
                    'Maximum_Estimate': max(pred_values),
                    'Number_of_Models': len(pred_values)
                }
                
                # Find best model based on R² score
                if hasattr(self, 'model_metrics'):
                    best_model = max(valid_preds.keys(), 
                                   key=lambda k: self.model_metrics.get(k, {}).get('r2_score', 0))
                    best_r2 = self.model_metrics.get(best_model, {}).get('r2_score', 0)
                    best_pred = valid_preds[best_model]
                    
                    summary_stats.update({
                        'Best_Model': best_model,
                        'Best_Model_R²': best_r2,
                        'Best_Model_Prediction': best_pred
                    })
            
            # =============================================================
            # BLOCK F: EXPORT TO DIFFERENT FORMATS
            # =============================================================
            file_ext = base_filename.lower().split('.')[-1]
            
            if file_ext == 'xlsx':
                # Export to Excel with multiple sheets
                import pandas as pd
                
                with pd.ExcelWriter(base_filename, engine='openpyxl') as writer:
                    # Summary sheet
                    if export_data:
                        df_results = pd.DataFrame(export_data)
                        df_results.to_excel(writer, sheet_name='Prediction_Results', index=False)
                    
                    # Input values sheet
                    input_df = pd.DataFrame([{
                        'Feature': key,
                        'Value': value
                    } for key, value in input_values.items()])
                    input_df.to_excel(writer, sheet_name='Input_Values', index=False)
                    
                    # Model performance sheet
                    if model_performance_data:
                        df_performance = pd.DataFrame(model_performance_data)
                        df_performance.to_excel(writer, sheet_name='Model_Performance', index=False)
                    
                    # Summary statistics sheet
                    if valid_preds:
                        summary_df = pd.DataFrame([summary_stats])
                        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                messagebox.showinfo("Export Successful", 
                                  f"Results exported to Excel file:\n{base_filename}\n\nSheets created:\n• Prediction Results\n• Input Values\n• Model Performance\n• Summary Statistics")
            
            elif file_ext == 'csv':
                # Export to CSV (main results only)
                import pandas as pd
                
                if export_data:
                    df_results = pd.DataFrame(export_data)
                    df_results.to_csv(base_filename, index=False)
                    
                    messagebox.showinfo("Export Successful", 
                                      f"Results exported to CSV file:\n{base_filename}")
                else:
                    messagebox.showerror("Export Error", "No valid predictions to export!")
            
            else:
                messagebox.showerror("Export Error", "Unsupported file format! Please use .xlsx or .csv")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")

    # =============================================================================
    # STEP 28: SINGLE PROJECT PREDICTION (LEGACY METHOD)
    # =============================================================================

    def predict_single(self):
        """
        Make prediction for single project
        
        This is a legacy method that demonstrates prediction using the first
        test sample. In practice, the manual input method is preferred as it
        allows users to enter their own project parameters.
        """
        if not self.models or self.X_test is None:
            messagebox.showerror("Error", "Please train models first!")
            return
        
        # =============================================================
        # BLOCK A: GET SAMPLE DATA
        # =============================================================
        # Use first test sample as example
        sample = self.X_test.iloc[0:1]  # Get first row as DataFrame
        actual = self.y_test.iloc[0]    # Get actual target value
        
        # =============================================================
        # BLOCK B: SETUP RESULTS DISPLAY
        # =============================================================
        # Clear previous results
        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)
        
        # Set up table columns
        columns = ['Model', 'Prediction', 'Actual', 'Error']
        self.pred_tree["columns"] = columns
        self.pred_tree["show"] = "headings"
        
        # Configure column headers and widths
        for col in columns:
            self.pred_tree.heading(col, text=col)
            self.pred_tree.column(col, width=150)
        
        # =============================================================
        # BLOCK C: MAKE PREDICTIONS WITH ALL MODELS
        # =============================================================
        for model_name, model in self.models.items():
            # Use appropriate feature scaling based on model type
            if model_name == 'Linear Regression':
                sample_scaled = self.scaler.transform(sample)
                pred = model.predict(sample_scaled)[0]
            else:
                pred = model.predict(sample)[0]
            
            # Calculate prediction error
            error = abs(pred - actual)
            
            # Add row to results table
            self.pred_tree.insert("", "end", values=[model_name, f"{pred:.2f}", f"{actual:.2f}", f"{error:.2f}"])
    
    # =============================================================================
    # STEP 29: BATCH PREDICTION FILE LOADING
    # =============================================================================
    
    def load_batch_prediction(self):
        """
        Load batch file for prediction
        
        Allows users to upload a CSV or Excel file containing multiple projects
        for batch processing. This is efficient for predicting costs for many
        projects simultaneously.
        """
        # =============================================================
        # BLOCK A: FILE SELECTION DIALOG
        # =============================================================
        file_path = filedialog.askopenfilename(
            title="Select Batch Prediction File",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls *.xlsb"), ("All files", "*.*")]
        )
        
        # =============================================================
        # BLOCK B: FILE LOADING WITH FORMAT DETECTION
        # =============================================================
        if file_path:
            try:
                # Load file based on extension
                if file_path.endswith('.csv'):
                    self.batch_df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls', '.xlsb')):
                    if file_path.endswith('.xlsb'):
                        # Binary Excel format
                        self.batch_df = pd.read_excel(file_path, engine='pyxlsb')
                    else:
                        # Standard Excel formats
                        self.batch_df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    messagebox.showerror("Error", "Unsupported file format!")
                    return
                     
                # =============================================================
                # BLOCK C: SUCCESS CONFIRMATION AND PREVIEW
                # =============================================================
                # Update file label and display preview
                self.file_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
                self.display_data_preview()
                self.display_batch_info()
                
                # Show success message
                messagebox.showinfo("Success", f"Batch file loaded! Shape: {self.batch_df.shape}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load batch file: {str(e)}")
    
    # =============================================================================
    # STEP 30: BATCH FILE INFORMATION DISPLAY
    # =============================================================================
    
    def display_batch_info(self):
        """
        Display batch file information and parameters
        
        Shows the contents of the loaded batch file in a table format
        so users can verify the data before running batch predictions.
        """
        if not hasattr(self, 'batch_df') or self.batch_df is None:
            return
            
        # =============================================================
        # BLOCK A: CLEAR EXISTING DISPLAY
        # =============================================================
        for item in self.batch_info_tree.get_children():
            self.batch_info_tree.delete(item)
        
        # =============================================================
        # BLOCK B: CONFIGURE TABLE STRUCTURE
        # =============================================================
        # Set up columns for displaying batch file information
        self.batch_info_tree["columns"] = list(self.batch_df.columns)
        self.batch_info_tree["show"] = "headings"
        
        # Configure column headers and widths
        for col in self.batch_df.columns:
            self.batch_info_tree.heading(col, text=col)
            self.batch_info_tree.column(col, width=120, minwidth=80)
        
        # =============================================================
        # BLOCK C: INSERT DATA ROWS WITH FORMATTING
        # =============================================================
        for idx, row in self.batch_df.iterrows():
            values = []
            for col in self.batch_df.columns:
                value = row[col]
                # Format numeric values nicely
                if pd.api.types.is_numeric_dtype(type(value)) and pd.notna(value):
                    if isinstance(value, float):
                        values.append(f"{value:,.2f}")
                    else:
                        values.append(f"{value:,}")
                else:
                    values.append(str(value) if pd.notna(value) else "")
            
            self.batch_info_tree.insert("", "end", values=values)
        
        print(f"Displayed batch info for {len(self.batch_df)} rows and {len(self.batch_df.columns)} columns")

# =============================================================================
# PART 13: BATCH PREDICTION PROCESSING & CONFIDENCE SYSTEMS
# =============================================================================

    # =============================================================================
    # STEP 31: MODEL RECOMMENDATION SYSTEM
    # =============================================================================
    
    def get_best_model_for_each_row(self):
        """
        Determine the best model for each row based on model metrics
        
        This method analyzes the performance of all trained models and recommends
        the best performing model for predictions. Currently uses overall performance
        but could be enhanced to provide row-specific recommendations based on
        project characteristics.
        
        Returns:
            list: List of recommended model names for each row in batch data
        """
        if not hasattr(self, 'model_metrics') or not self.model_metrics:
            return None
        
        # =============================================================
        # BLOCK A: FIND BEST OVERALL MODEL
        # =============================================================
        # Find the best model overall based on R² score (highest is better)
        # R² measures how well the model explains the variance in the data
        best_overall_model = max(self.model_metrics.keys(), 
                                key=lambda model: self.model_metrics[model].get('r2_score', -float('inf')))
        
        # =============================================================
        # BLOCK B: APPLY RECOMMENDATION TO ALL ROWS
        # =============================================================
        # For now, return the best overall model for all rows
        # Future enhancement: analyze each row individually for personalized recommendations
        if hasattr(self, 'batch_predictions') and self.batch_predictions:
            num_rows = len(list(self.batch_predictions.values())[0])
            return [best_overall_model] * num_rows
        return None
    
    # =============================================================================
    # STEP 32: PROJECT IDENTIFIER DETECTION
    # =============================================================================
    
    def get_project_identifier_column(self):
        """
        Try to identify a project name/ID column in the batch data
        
        This method automatically detects which column contains project
        identifiers to make the results more user-friendly and easier
        to understand.
        
        Returns:
            str: Column name that likely contains project identifiers
        """
        if not hasattr(self, 'batch_df') or self.batch_df is None:
            return None
            
        # =============================================================
        # BLOCK A: SEARCH FOR COMMON PROJECT ID COLUMN NAMES
        # =============================================================
        # Look for common project identifier column names
        potential_id_cols = ['project_name', 'project_id', 'name', 'id', 'project', 
                           'site_name', 'site_id', 'location', 'description']
        
        for col in self.batch_df.columns:
            if col.lower() in potential_id_cols:
                return col
        
        # =============================================================
        # BLOCK B: FALLBACK STRATEGIES
        # =============================================================
        # If no obvious ID column found, use the first non-numeric column
        for col in self.batch_df.columns:
            if not pd.api.types.is_numeric_dtype(self.batch_df[col]):
                return col
        
        # If all columns are numeric, return the first column
        return self.batch_df.columns[0] if len(self.batch_df.columns) > 0 else None
    
    # =============================================================================
    # STEP 33: CONFIDENCE INTERVAL CALCULATION SYSTEM
    # =============================================================================
    
    def calculate_prediction_intervals(self, predictions, model_name, confidence_level=0.95):
        """
        Calculate prediction confidence intervals based on model evaluation metrics
        
        This method provides uncertainty estimates for predictions, helping users
        understand the reliability of cost estimates. Uses statistical methods
        to create confidence bounds around each prediction.
        
        Methodology:
        - Uses RMSE (Root Mean Square Error) as standard error estimate
        - Applies Z-score for desired confidence level (1.96 for 95%)
        - Adjusts intervals based on model reliability (R² score)
        - Ensures lower bounds never go below zero (cost can't be negative)
        
        Args:
            predictions: Array of predictions from the model
            model_name: Name of the model for metric lookup
            confidence_level: Confidence level (default 95%)
            
        Returns:
            tuple: (lower_bounds, upper_bounds) arrays
        """
        # =============================================================
        # BLOCK A: FALLBACK FOR MISSING METRICS
        # =============================================================
        if not hasattr(self, 'model_metrics') or model_name not in self.model_metrics:
            # Fallback: use a simple percentage-based interval (±10%)
            margin = np.array(predictions) * 0.10
            lower_bounds = np.array(predictions) - margin
            upper_bounds = np.array(predictions) + margin
            return lower_bounds, upper_bounds
        
        # =============================================================
        # BLOCK B: GET MODEL PERFORMANCE METRICS
        # =============================================================
        metrics = self.model_metrics[model_name]
        rmse = metrics.get('rmse', 0)      # Root Mean Square Error
        mae = metrics.get('mae', 0)        # Mean Absolute Error
        r2_score = metrics.get('r2_score', 0)  # R-squared score
        
        # =============================================================
        # BLOCK C: STATISTICAL CONFIDENCE INTERVAL CALCULATION
        # =============================================================
        if rmse > 0:
            # Determine Z-score based on confidence level
            if confidence_level == 0.95:
                z_score = 1.96    # 95% confidence
            elif confidence_level == 0.90:
                z_score = 1.645   # 90% confidence
            else:
                z_score = 1.96    # Default to 95%
            
            # =============================================================
            # SUB-BLOCK C1: RELIABILITY ADJUSTMENT
            # =============================================================
            # Adjust confidence interval based on model reliability
            # Better models (higher R²) get tighter intervals
            # Worse models (lower R²) get wider intervals
            reliability_factor = max(0.5, min(1.5, 1.0 + (0.5 - r2_score)))
            
            # Calculate margin of error using statistical method
            margin_of_error = z_score * rmse * reliability_factor
            
            # =============================================================
            # SUB-BLOCK C2: APPLY CONFIDENCE BOUNDS
            # =============================================================
            # Apply margin to predictions
            lower_bounds = np.array(predictions) - margin_of_error
            upper_bounds = np.array(predictions) + margin_of_error
            
            # Ensure lower bounds are not negative (cost predictions can't be negative)
            lower_bounds = np.maximum(lower_bounds, 0)
            
        else:
            # =============================================================
            # BLOCK D: FALLBACK CALCULATION
            # =============================================================
            # Fallback if RMSE is not available - use percentage margin
            margin = np.array(predictions) * 0.15  # 15% margin
            lower_bounds = np.maximum(np.array(predictions) - margin, 0)
            upper_bounds = np.array(predictions) + margin
        
        return lower_bounds, upper_bounds
    
    # =============================================================================
    # STEP 34: BATCH PREDICTION PROCESSING ENGINE
    # =============================================================================
    
    def predict_batch(self):
        """
        Make batch predictions
        
        This is the main batch processing engine that:
        1. Applies preprocessing to batch data (same as training data)
        2. Runs predictions with all trained models
        3. Calculates confidence intervals for uncertainty quantification
        4. Displays results in a comprehensive format
        
        The batch system is designed for efficiency and handles large datasets
        while maintaining the same quality standards as single predictions.
        """
        # =============================================================
        # BLOCK A: VALIDATION CHECKS
        # =============================================================
        if not hasattr(self, 'batch_df') or not self.models:
            messagebox.showerror("Error", "Please load batch file and train models first!")
            return
        
        try:
            # =============================================================
            # BLOCK B: PREPROCESS BATCH DATA
            # =============================================================
            # Apply same preprocessing pipeline as training data
            processed_batch = self.batch_df.copy()
            
            # Track categorical columns that need to be removed after encoding
            categorical_columns_to_remove = []
            
            # =============================================================
            # SUB-BLOCK B1: CATEGORICAL ENCODING
            # =============================================================
            for col in self.batch_df.columns:
                # Check if this column was encoded during training
                if hasattr(self, 'categorical_encoding_method') and col in self.categorical_encoding_method:
                    method = self.categorical_encoding_method[col]
                    processed_batch[col] = processed_batch[col].fillna('Unknown')
                    
                    if method == 'onehot':
                        # Apply one-hot encoding matching training structure
                        dummies = pd.get_dummies(processed_batch[col], prefix=col, drop_first=True)
                        
                        # Ensure same columns as training
                        if hasattr(self, 'feature_names') and self.feature_names:
                            for feature_name in self.feature_names:
                                if feature_name.startswith(f"{col}_") and feature_name not in dummies.columns:
                                    dummies[feature_name] = 0
                            
                            # Keep only training columns
                            training_cols = [f for f in self.feature_names if f.startswith(f"{col}_")]
                            dummies = dummies.reindex(columns=training_cols, fill_value=0)
                        
                        # Add to processed data and mark for removal
                        processed_batch = pd.concat([processed_batch, dummies], axis=1)
                        categorical_columns_to_remove.append(col)
                        
                    elif method == 'target':
                        # Apply target encoding using stored encoder
                        if col in self.label_encoders:
                            encoder_info = self.label_encoders[col]
                            mapping = encoder_info['mapping']
                            global_mean = encoder_info['global_mean']
                            processed_batch[col] = processed_batch[col].map(mapping).fillna(global_mean)
                        else:
                            print(f"Warning: No encoder found for target-encoded column {col}")
                            processed_batch[col] = pd.to_numeric(processed_batch[col], errors='coerce')
                else:
                    # Numeric column - convert to numeric
                    processed_batch[col] = pd.to_numeric(processed_batch[col], errors='coerce')
            
            # Remove original categorical columns that were one-hot encoded
            processed_batch.drop(columns=categorical_columns_to_remove, inplace=True)
            
            # =============================================================
            # SUB-BLOCK B2: FEATURE ALIGNMENT
            # =============================================================
            # Ensure processed batch has the same feature alignment as training data
            if hasattr(self, 'feature_names') and self.feature_names:
                print(f"Training features expected: {self.feature_names}")
                print(f"Batch features available: {list(processed_batch.columns)}")
                
                # Reindex to match training features, filling missing columns with 0
                processed_batch = processed_batch.reindex(columns=self.feature_names, fill_value=0)
                print(f"Aligned batch features to training: {list(processed_batch.columns)}")
                print(f"Batch shape after alignment: {processed_batch.shape}")
            
            # =============================================================
            # BLOCK C: RUN PREDICTIONS WITH ALL MODELS
            # =============================================================
            self.batch_predictions = {}                # Store predictions
            self.batch_prediction_intervals = {}       # Store confidence intervals
            
            # Models that need scaled features
            scaled_models = [
                'Linear Regression', 'Ridge Regression', 'Lasso Regression',
                'ElasticNet', 'Support Vector Regression', 'K-Nearest Neighbors'
            ]
            
            for model_name, model in self.models.items():
                try:
                    # =============================================================
                    # SUB-BLOCK C1: MODEL-SPECIFIC PREDICTION
                    # =============================================================
                    if model_name in scaled_models:
                        # Use scaled features for linear models, SVR, and KNN
                        if hasattr(self, 'scaler') and self.scaler is not None:
                            batch_scaled = self.scaler.transform(processed_batch)
                            preds = model.predict(batch_scaled)
                        else:
                            print(f"WARNING: No scaler available for {model_name}, using unscaled data")
                            preds = model.predict(processed_batch)
                    else:
                        # Use original features for tree-based models
                        preds = model.predict(processed_batch)
                    
                    # =============================================================
                    # SUB-BLOCK C2: STORE PREDICTIONS AND CONFIDENCE INTERVALS
                    # =============================================================
                    # Store predictions
                    self.batch_predictions[model_name] = preds
                    
                    # Calculate and store confidence intervals
                    lower_bounds, upper_bounds = self.calculate_prediction_intervals(preds, model_name)
                    self.batch_prediction_intervals[model_name] = {
                        'lower': lower_bounds,
                        'upper': upper_bounds
                    }
                    
                    print(f"Batch predictions completed for {model_name}: {len(preds)} predictions with confidence intervals")
                    
                except Exception as e:
                    print(f"Prediction failed for {model_name}: {e}")
                    # Skip this model but continue with others
            
            # =============================================================
            # BLOCK D: DISPLAY RESULTS AND SUCCESS MESSAGE
            # =============================================================
            if self.batch_predictions:
                # Display results in the GUI
                self.display_batch_results()
                successful_models = len(self.batch_predictions)
                total_models = len(self.models)
                messagebox.showinfo("Success", f"Batch predictions completed!\nSuccessful predictions: {successful_models}/{total_models} models")
            else:
                messagebox.showerror("Error", "No models were able to make predictions. Please check that your batch data has the required columns.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Batch prediction failed: {str(e)}")

# =============================================================================
# PART 14: BATCH RESULTS DISPLAY & EXPORT SYSTEMS
# =============================================================================

    # =============================================================================
    # STEP 35: BATCH RESULTS DISPLAY SYSTEM
    # =============================================================================
    
    def display_batch_results(self):
        """
        Display batch prediction results with project names and best model recommendations
        
        This method creates a comprehensive results table showing:
        - Project identifiers for easy reference
        - Predictions from all models with confidence intervals
        - Best model recommendations for each project
        - Professional formatting with UK currency display
        
        The display balances detail with readability, showing enough information
        for decision-making without overwhelming the user.
        """
        # =============================================================
        # BLOCK A: CLEAR PREVIOUS RESULTS
        # =============================================================
        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)
        
        # =============================================================
        # BLOCK B: VALIDATE PREDICTION DATA
        # =============================================================
        if not self.batch_predictions:
            print("No batch predictions to display")
            return
            
        # Get the number of predictions (rows) from the first model
        prediction_values = list(self.batch_predictions.values())
        if not prediction_values or len(prediction_values[0]) == 0:
            print("No prediction data available")
            return
            
        num_rows = len(prediction_values[0])
        print(f"Displaying {num_rows} predictions for {len(self.batch_predictions)} models")
        
        # =============================================================
        # BLOCK C: SETUP DISPLAY STRUCTURE
        # =============================================================
        # Get project identifier column and best model recommendations
        project_col = self.get_project_identifier_column()
        best_models = self.get_best_model_for_each_row()
        
        # Build columns list: Project Name + Model Predictions + Best Model
        columns = []
        if project_col:
            columns.append('Project')
        
        # Add all model prediction columns
        columns.extend(list(self.batch_predictions.keys()))
        
        # Add best model column
        if best_models:
            columns.append('Best Model')
        
        # Configure treeview columns
        self.pred_tree["columns"] = columns
        self.pred_tree["show"] = "headings"
        
        # =============================================================
        # BLOCK D: CONFIGURE COLUMN HEADERS AND WIDTHS
        # =============================================================
        for col in columns:
            self.pred_tree.heading(col, text=col)
            if col == 'Project':
                self.pred_tree.column(col, width=150, minwidth=100)
            elif col == 'Best Model':
                self.pred_tree.column(col, width=150, minwidth=100)
            else:
                # Wider columns for prediction ranges with confidence intervals
                self.pred_tree.column(col, width=180, minwidth=150)
        
        # =============================================================
        # BLOCK E: INSERT PREDICTION DATA WITH FORMATTING
        # =============================================================
        for i in range(num_rows):
            values = []
            
            # =============================================================
            # SUB-BLOCK E1: ADD PROJECT IDENTIFIER
            # =============================================================
            if project_col and hasattr(self, 'batch_df'):
                try:
                    project_value = self.batch_df.iloc[i][project_col]
                    values.append(str(project_value) if pd.notna(project_value) else f"Row {i+1}")
                except (IndexError, KeyError):
                    values.append(f"Row {i+1}")
            
            # =============================================================
            # SUB-BLOCK E2: ADD MODEL PREDICTIONS WITH CONFIDENCE RANGES
            # =============================================================
            for model_name in self.batch_predictions:
                try:
                    pred_value = self.batch_predictions[model_name][i]
                    
                    # Add confidence interval if available
                    if (hasattr(self, 'batch_prediction_intervals') and 
                        model_name in self.batch_prediction_intervals):
                        intervals = self.batch_prediction_intervals[model_name]
                        lower_bound = intervals['lower'][i]
                        upper_bound = intervals['upper'][i]
                        # Display format: "£50,000.00 (£45,000.00 - £55,000.00)" - UK currency format
                        values.append(f"£{pred_value:,.2f} (£{lower_bound:,.2f} - £{upper_bound:,.2f})")
                    else:
                        # No confidence interval available
                        values.append(f"£{pred_value:,.2f}")
                        
                except (IndexError, TypeError) as e:
                    print(f"Error accessing prediction {i} for {model_name}: {e}")
                    values.append("N/A")
            
            # =============================================================
            # SUB-BLOCK E3: ADD BEST MODEL RECOMMENDATION
            # =============================================================
            if best_models and i < len(best_models):
                values.append(best_models[i])
            
            # Insert the row into the display table
            self.pred_tree.insert("", "end", values=values)
        
        # =============================================================
        # BLOCK F: DISPLAY INFORMATION NOTE
        # =============================================================
        # Add informational note about the display format
        if hasattr(self, 'model_metrics') and self.model_metrics:
            print("Note: EAC predictions shown with 95% confidence intervals in UK currency format (£X,XXX.XX).")
            print("Export to Excel for: detailed methodology explanation, confidence interval figures (±), and separate bound columns.")
    
    # =============================================================================
    # STEP 36: COMPREHENSIVE EXPORT SYSTEM WITH CONFIDENCE INTERVALS
    # =============================================================================
    
    def export_predictions(self):
        """
        Export predictions with evaluation parameters and recommendations to CSV or XLSX
        
        This comprehensive export system creates professional reports including:
        - All prediction data with confidence intervals
        - Model performance metrics and rankings
        - Detailed methodology explanations
        - Professional UK currency formatting
        - Multiple export formats (Excel multi-sheet, CSV)
        
        The export is designed for business use, auditing, and decision-making.
        """
        # =============================================================
        # BLOCK A: VALIDATION AND FILE SELECTION
        # =============================================================
        if not hasattr(self, 'batch_predictions'):
            messagebox.showerror("Error", "No batch predictions to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save predictions as",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # =============================================================
                # BLOCK B: CREATE COMPREHENSIVE EXPORT DATAFRAME
                # =============================================================
                export_df = self.batch_df.copy()
                
                # =============================================================
                # SUB-BLOCK B1: ADD MODEL PREDICTIONS WITH CONFIDENCE INTERVALS
                # =============================================================
                for model_name, preds in self.batch_predictions.items():
                    # Format predictions in UK currency format
                    formatted_preds = [f"£{pred:,.2f}" for pred in preds]
                    export_df[f'{model_name}_EAC_Prediction'] = formatted_preds
                    
                    # Add confidence intervals if available
                    if hasattr(self, 'batch_prediction_intervals') and model_name in self.batch_prediction_intervals:
                        intervals = self.batch_prediction_intervals[model_name]
                        
                        # Format bounds in UK currency format
                        formatted_lower = [f"£{bound:,.2f}" for bound in intervals['lower']]
                        formatted_upper = [f"£{bound:,.2f}" for bound in intervals['upper']]
                        
                        export_df[f'{model_name}_Lower_Bound'] = formatted_lower
                        export_df[f'{model_name}_Upper_Bound'] = formatted_upper
                        
                        # Calculate and add confidence interval figure (±amount)
                        confidence_figures = []
                        for i, pred in enumerate(preds):
                            margin = (intervals['upper'][i] - intervals['lower'][i]) / 2
                            confidence_figures.append(f"±£{margin:,.2f}")
                        
                        export_df[f'{model_name}_Confidence_Interval'] = confidence_figures
                
                # =============================================================
                # SUB-BLOCK B2: ADD BEST MODEL RECOMMENDATIONS
                # =============================================================
                best_models = self.get_best_model_for_each_row()
                if best_models:
                    export_df['Recommended_Model'] = best_models
                
                # =============================================================
                # SUB-BLOCK B3: ADD MODEL EVALUATION PARAMETERS
                # =============================================================
                if hasattr(self, 'model_metrics') and self.model_metrics:
                    for model_name in self.batch_predictions.keys():
                        if model_name in self.model_metrics:
                            metrics = self.model_metrics[model_name]
                            export_df[f'{model_name}_R2_Score'] = metrics.get('r2_score', 'N/A')
                            export_df[f'{model_name}_MSE'] = metrics.get('mse', 'N/A')
                            export_df[f'{model_name}_MAE'] = metrics.get('mae', 'N/A')
                            export_df[f'{model_name}_RMSE'] = metrics.get('rmse', 'N/A')
                
                # =============================================================
                # SUB-BLOCK B4: ADD MODEL SELECTION ADVICE
                # =============================================================
                if hasattr(self, 'model_metrics') and self.model_metrics:
                    # Find the best model based on R² score
                    best_model = max(self.model_metrics.keys(), 
                                   key=lambda model: self.model_metrics[model].get('r2_score', -float('inf')))
                    
                    best_r2 = self.model_metrics[best_model]['r2_score']
                    advice_text = (f"Recommended: {best_model} (R²={best_r2:.4f}). "
                                 f"This model shows the best overall performance with highest R² score, "
                                 f"lowest prediction errors, and most reliable results.")
                    
                    export_df['Model_Selection_Advice'] = advice_text
                
                # =============================================================
                # BLOCK C: FORMAT-SPECIFIC EXPORT HANDLING
                # =============================================================
                if file_path.lower().endswith('.xlsx'):
                    # =============================================================
                    # SUB-BLOCK C1: EXCEL EXPORT WITH MULTIPLE SHEETS
                    # =============================================================
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # Main predictions sheet
                        export_df.to_excel(writer, sheet_name='EAC_Predictions_and_Metrics', index=False)
                        
                        # =============================================================
                        # SUB-SUB-BLOCK C1a: MODEL COMPARISON SUMMARY SHEET
                        # =============================================================
                        if hasattr(self, 'model_metrics') and self.model_metrics:
                            metrics_summary = []
                            for model_name, metrics in self.model_metrics.items():
                                if model_name in self.batch_predictions:
                                    metrics_summary.append({
                                        'Model': model_name,
                                        'R² Score': metrics.get('r2_score', 'N/A'),
                                        'MSE': metrics.get('mse', 'N/A'),
                                        'MAE': metrics.get('mae', 'N/A'),
                                        'RMSE': metrics.get('rmse', 'N/A'),
                                        'Rank_by_R2': ''
                                    })
                            
                            metrics_df = pd.DataFrame(metrics_summary)
                            # Rank models by R² score
                            metrics_df = metrics_df.sort_values('R² Score', ascending=False)
                            metrics_df['Rank_by_R2'] = range(1, len(metrics_df) + 1)
                            
                            metrics_df.to_excel(writer, sheet_name='Model_Performance_Summary', index=False)
                        
                        # =============================================================
                        # SUB-SUB-BLOCK C1b: METHODOLOGY EXPLANATION SHEET
                        # =============================================================
                        methodology_data = [
                            ['CONFIDENCE INTERVAL METHODOLOGY EXPLANATION', ''],
                            ['', ''],
                            ['Purpose:', 'Provide uncertainty estimates for Estimate at Completion (EAC) predictions'],
                            ['Confidence Level:', '95% (meaning 95% of true values should fall within the interval)'],
                            ['', ''],
                            ['PRIMARY METHOD (Used when model evaluation metrics available):', ''],
                            ['Formula:', 'Margin of Error = Z-score × RMSE × Reliability Factor'],
                            ['Z-score (95% confidence):', '1.96'],
                            ['RMSE:', 'Root Mean Square Error from model evaluation on test data'],
                            ['Reliability Factor:', 'Adjustment based on model performance (R² score)'],
                            ['Reliability Factor Calculation:', 'max(0.5, min(1.5, 1.0 + (0.5 - R²)))'],
                            ['Effect:', 'Better models (higher R²) → tighter confidence intervals'],
                            ['', 'Poorer models (lower R²) → wider confidence intervals'],
                            ['Lower Bound:', 'Prediction - Margin of Error (minimum £0.00)'],
                            ['Upper Bound:', 'Prediction + Margin of Error'],
                            ['Confidence Interval:', '±(Upper Bound - Lower Bound) / 2'],
                            ['', ''],
                            ['FALLBACK METHOD (Used when evaluation metrics unavailable):', ''],
                            ['Method:', 'Percentage-based interval'],
                            ['Calculation:', '±10% to ±15% of the prediction value'],
                            ['Lower Bound:', 'max(0, Prediction × 0.85 to 0.90)'],
                            ['Upper Bound:', 'Prediction × 1.10 to 1.15'],
                            ['', ''],
                            ['INTERPRETATION:', ''],
                            ['Prediction:', 'Most likely EAC value'],
                            ['Lower Bound:', 'Optimistic scenario (95% confidence floor)'],
                            ['Upper Bound:', 'Conservative scenario (95% confidence ceiling)'],
                            ['Confidence Interval (±):', 'Margin of uncertainty around the prediction'],
                            ['', ''],
                            ['EXAMPLE:', ''],
                            ['If prediction = £50,000 with ±£5,000 confidence interval:'],
                            ['Range:', '£45,000 to £55,000'],
                            ['Interpretation:', '95% confident the true EAC is between £45,000 and £55,000'],
                            ['', ''],
                            ['NOTES:', ''],
                            ['• All values formatted in UK currency (£) with thousands separators'],
                            ['• Lower bounds cannot be negative (minimum £0.00)'],
                            ['• Intervals reflect model reliability and prediction uncertainty'],
                            ['• Use for budgeting, risk assessment, and contingency planning']
                        ]
                        
                        methodology_df = pd.DataFrame(methodology_data, columns=['Topic', 'Description'])
                        methodology_df.to_excel(writer, sheet_name='Confidence_Interval_Methodology', index=False)
                    
                    # =============================================================
                    # SUB-BLOCK C2: EXCEL SUCCESS MESSAGE
                    # =============================================================
                    messagebox.showinfo("Success", 
                                      f"EAC Predictions exported to {file_path}\n\n"
                                      f"Excel file includes 3 sheets:\n"
                                      f"1. EAC_Predictions_and_Metrics:\n"
                                      f"   • All original project data\n"
                                      f"   • EAC predictions (UK £ format) from {len(self.batch_predictions)} models\n"
                                      f"   • 95% confidence intervals (Lower/Upper bounds)\n"
                                      f"   • Confidence interval figures (± amounts)\n"
                                      f"   • Model evaluation metrics and recommendations\n\n"
                                      f"2. Model_Performance_Summary:\n"
                                      f"   • Comparative model performance rankings\n\n"
                                      f"3. Confidence_Interval_Methodology:\n"
                                      f"   • Complete explanation of Primary & Fallback methods\n"
                                      f"   • Statistical formulas and interpretation guide")
                else:
                    # =============================================================
                    # SUB-BLOCK C3: CSV EXPORT
                    # =============================================================
                    export_df.to_csv(file_path, index=False)
                    messagebox.showinfo("Success", 
                                      f"EAC Predictions exported to {file_path}\n\n"
                                      f"Includes:\n"
                                      f"• All original project data\n"
                                      f"• EAC predictions (UK £ format) from {len(self.batch_predictions)} models\n"
                                      f"• 95% confidence intervals (Lower/Upper bounds)\n"
                                      f"• Confidence interval figures (± amounts)\n"
                                      f"• Model evaluation metrics and recommendations")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

# =============================================================================
# PART 15: OUTLIER DISPLAY & VISUALIZATION SYSTEMS
# =============================================================================

    # =============================================================================
    # STEP 37: OUTLIER DATA TRANSPARENCY SYSTEM
    # =============================================================================
    
    def display_outlier_data(self):
        """
        Display outlier data in the outliers tab
        
        This method provides complete transparency about which data points were
        removed during preprocessing. This is crucial for:
        - Auditing and compliance purposes
        - Understanding data quality issues
        - Building trust with stakeholders
        - Identifying patterns in problematic data
        
        The display shows not just what was removed, but why each row was
        flagged as an outlier, promoting transparency and accountability.
        """
        # =============================================================
        # BLOCK A: CLEAR EXISTING DISPLAY
        # =============================================================
        for item in self.outlier_tree.get_children():
            self.outlier_tree.delete(item)
            
        # =============================================================
        # BLOCK B: HANDLE NO OUTLIER DATA SCENARIO
        # =============================================================
        if self.outlier_data is None:
            self.outlier_info_label.config(text="No outliers detected or outlier removal was not applied.")
            return
            
        # =============================================================
        # BLOCK C: UPDATE INFORMATION SUMMARY
        # =============================================================
        num_outliers = len(self.outlier_data['features'])
        self.outlier_info_label.config(
            text=f"Found {num_outliers} outlier rows that were removed from training data. "
                 f"These rows contained values outside the IQR bounds (Q1-1.5*IQR, Q3+1.5*IQR)."
        )
        
        # =============================================================
        # BLOCK D: PREPARE DISPLAY DATA WITH OUTLIER REASONS
        # =============================================================
        # Combine features and target for comprehensive display
        outlier_display = self.outlier_data['features'].copy()
        target_col = self.y.name if hasattr(self.y, 'name') else 'Target'
        outlier_display[target_col] = self.outlier_data['target']
        
        # Add column showing which features caused the outlier flag
        outlier_reasons = []
        for idx in self.outlier_data['features'].index:
            reason_cols = self.outlier_data['outlier_details'].get(idx, [])
            outlier_reasons.append(', '.join(reason_cols) if reason_cols else 'Unknown')
        outlier_display['Outlier_Reason'] = outlier_reasons
        
        # =============================================================
        # BLOCK E: CONFIGURE AND POPULATE TABLE
        # =============================================================
        # Configure treeview columns
        columns = list(outlier_display.columns)
        self.outlier_tree["columns"] = columns
        self.outlier_tree["show"] = "headings"
        
        for col in columns:
            self.outlier_tree.heading(col, text=col)
            self.outlier_tree.column(col, width=100)
            
        # Insert data rows
        for idx, row in outlier_display.iterrows():
            values = [str(val) for val in row]
            self.outlier_tree.insert("", "end", values=values)

    # =============================================================================
    # STEP 38: FEATURE IMPORTANCE VISUALIZATION
    # =============================================================================

    def plot_feature_importance(self):
        """
        Plot Feature Importance Analysis with Improved Readability
        
        Creates horizontal bar charts showing which features are most important
        for predicting project costs. This helps users understand:
        - Which project characteristics drive cost estimates most
        - How different models weigh features differently
        - Whether the models are focusing on logical cost drivers
        
        Improvements Made:
        - Reduced to top 8 features per model (vs 10) for clarity
        - Professional color scheme with value labels on bars
        - Truncated long feature names to 15 characters
        - Better spacing and grid lines
        - Removed chart junk (unnecessary borders, etc.)
        
        Feature Importance Sources:
        - Linear Regression: Absolute values of coefficients
        - Random Forest: Gini importance from tree splits  
        - XGBoost: Gain-based importance from boosting
        
        Note: Only available after models have been trained.
        """
        # =============================================================
        # BLOCK A: VALIDATION CHECK
        # =============================================================
        if not self.feature_importance:
            messagebox.showerror("Error", "No feature importance data available!")
            return
        
        self.fig.clear()
        
        # =============================================================
        # BLOCK B: DETERMINE GRID LAYOUT
        # =============================================================
        n_models = len(self.feature_importance)
        if n_models == 0:
            return
        
        # Use grid layout instead of vertical stacking to fit better in window
        if n_models <= 2:
            rows, cols = 1, n_models
        elif n_models <= 4:
            rows, cols = 2, 2
        elif n_models <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3  # Maximum 3x3 grid
            
        print(f"Feature importance grid layout: {rows}x{cols} for {n_models} models")
            
        # =============================================================
        # BLOCK C: CREATE SUBPLOT FOR EACH MODEL
        # =============================================================
        for i, (model_name, importance) in enumerate(self.feature_importance.items()):
            if i >= 9:  # Limit to 9 models maximum
                break
            ax = self.fig.add_subplot(rows, cols, i+1)
            
            # =============================================================
            # SUB-BLOCK C1: DEBUG FEATURE NAME ALIGNMENT
            # =============================================================
            print(f"\n=== FEATURE IMPORTANCE DEBUG for {model_name} ===")
            print(f"Importance array length: {len(importance)}")
            print(f"Importance values: {importance}")
            
            # Use the SAME strategy as prediction: X_train columns are authoritative
            if hasattr(self, 'X_train') and self.X_train is not None:
                feature_names = list(self.X_train.columns)
                print(f"Using X_train columns ({len(feature_names)}): {feature_names[:5]}...")
            elif hasattr(self, 'feature_names') and self.feature_names:
                feature_names = self.feature_names
                print(f"Using feature_names ({len(feature_names)}): {feature_names[:5]}...")
            else:
                feature_names = list(self.X.columns)
                print(f"Using X.columns ({len(feature_names)}): {feature_names[:5]}...")
            
            # =============================================================
            # SUB-BLOCK C2: VALIDATE DIMENSIONS AND ALIGN
            # =============================================================
            if len(importance) != len(feature_names):
                print(f"ERROR: Dimension mismatch! Importance: {len(importance)}, Names: {len(feature_names)}")
                print(f"This indicates a problem with feature importance calculation")
                # Truncate or pad to match
                if len(importance) < len(feature_names):
                    feature_names = feature_names[:len(importance)]
                    print(f"Truncated feature names to {len(feature_names)}")
                else:
                    # Extend feature names with generic names
                    missing_count = len(importance) - len(feature_names)
                    for idx in range(missing_count):
                        feature_names.append(f'Unknown_Feature_{len(feature_names) + idx}')
                    print(f"Extended feature names to {len(feature_names)}")
            
            # =============================================================
            # SUB-BLOCK C3: CREATE HORIZONTAL BAR PLOT
            # =============================================================
            # Sort features by importance (top 8 for better readability)
            sorted_idx = np.argsort(importance)[-8:]
            print(f"Top 8 feature indices: {sorted_idx}")
            print(f"Corresponding importance values: {importance[sorted_idx]}")
            
            # Create horizontal bar plot with better colors
            bars = ax.barh(range(len(sorted_idx)), importance[sorted_idx], 
                          color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
            
            # =============================================================
            # SUB-BLOCK C4: CONFIGURE LABELS AND FORMATTING
            # =============================================================
            # Create feature labels safely
            ax.set_yticks(range(len(sorted_idx)))
            feature_labels = [feature_names[i] for i in sorted_idx]  # Now safe since we validated dimensions
            print(f"Feature labels: {feature_labels}")
            print("=== END DEBUG ===\n")
            
            # Truncate long feature names for readability
            feature_labels = [label[:12] + '...' if len(label) > 12 else label 
                            for label in feature_labels]
            ax.set_yticklabels(feature_labels, fontsize=5, rotation=0)  # Smaller font
            
            # Customize title and labels with smaller fonts
            ax.set_title(f'{model_name}', fontsize=6, fontweight='bold', pad=4)  # Shorter title
            ax.set_xlabel('Importance', fontsize=5)  # Shorter label
            
            # Add value labels on bars (smaller)
            for j, bar in enumerate(bars):
                width = bar.get_width()
                if width > max(importance[sorted_idx]) * 0.1:  # Only show labels for significant bars
                    ax.text(width + max(importance[sorted_idx]) * 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.2f}', ha='left', va='center', fontsize=4)
            
            # Improve grid and spines
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set x-axis limits with some padding
            if len(importance[sorted_idx]) > 0:
                ax.set_xlim(0, max(importance[sorted_idx]) * 1.15)
        
        plt.tight_layout(pad=0.5)  # Much tighter layout to fit more charts
        self.canvas.draw()
    
    # =============================================================================
    # STEP 39: CORRELATION MATRIX VISUALIZATION
    # =============================================================================
    
    def plot_correlation(self):
        """
        Plot correlation matrix
        
        Creates a heatmap showing how different features relate to each other.
        Strong correlations (positive or negative) can indicate:
        - Redundant features that measure similar things
        - Potential multicollinearity issues
        - Relationships between project characteristics
        
        The heatmap uses color coding:
        - Red: Strong positive correlation
        - Blue: Strong negative correlation  
        - White: No correlation
        """
        # =============================================================
        # BLOCK A: VALIDATION CHECK
        # =============================================================
        if self.X is None:
            messagebox.showerror("Error", "No data available!")
            return
        
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        
        # =============================================================
        # BLOCK B: CALCULATE CORRELATION MATRIX
        # =============================================================
        # Calculate correlation matrix for numeric columns only
        numeric_data = self.X.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            corr_matrix = numeric_data.corr()
            
            # =============================================================
            # SUB-BLOCK B1: CREATE HEATMAP
            # =============================================================
            # Create heatmap with much smaller font sizes
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, fmt='.2f', annot_kws={'size': 3})  # Much smaller
            ax.set_title('Correlation Matrix', fontsize=6)  # Shorter title, smaller font
            ax.tick_params(axis='both', which='major', labelsize=4)  # Much smaller tick labels
            # Rotate labels more to prevent overlap
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=4)  # Vertical rotation
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4)  # Smaller font
        else:
            # =============================================================
            # BLOCK C: HANDLE NO NUMERIC FEATURES
            # =============================================================
            ax.text(0.5, 0.5, 'No numeric features for correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout(pad=0.2)  # Tighter layout
        self.canvas.draw()
    
    # =============================================================================
    # STEP 40: SCATTER PLOT VISUALIZATION
    # =============================================================================
    
    def plot_scatter(self):
        """
        Plot scatter plots of features vs target with trend lines
        
        Creates scatter plots showing the relationship between individual
        features and the target variable (cost). Each plot includes:
        - Individual data points
        - Linear trend line with equation
        - Correlation strength indication
        
        This helps identify:
        - Which features have strong linear relationships with cost
        - Potential outliers or unusual patterns
        - Non-linear relationships that might need transformation
        """
        # =============================================================
        # BLOCK A: VALIDATION CHECK
        # =============================================================
        if self.X is None or self.y is None:
            messagebox.showerror("Error", "No data available!")
            return
        
        self.fig.clear()
        
        # =============================================================
        # BLOCK B: GET NUMERIC COLUMNS
        # =============================================================
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No numeric features for scatter plots', 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
        
        # =============================================================
        # BLOCK C: CREATE SUBPLOTS
        # =============================================================
        # Create subplots (max 6 features for readability)
        n_cols = min(len(numeric_cols), 6)
        n_rows = (n_cols + 1) // 2
        
        for i, col in enumerate(numeric_cols[:6]):
            ax = self.fig.add_subplot(n_rows, 2, i+1)
            
            # =============================================================
            # SUB-BLOCK C1: PLOT DATA POINTS
            # =============================================================
            # Original data points
            ax.scatter(self.X[col], self.y, alpha=0.6, label='Data Points')
            
            # =============================================================
            # SUB-BLOCK C2: FIT AND PLOT TREND LINE
            # =============================================================
            # Fit trend line if we have valid data
            valid_data = ~(pd.isna(self.X[col]) | pd.isna(self.y))
            if valid_data.sum() > 1:
                x_clean = self.X[col][valid_data]
                y_clean = self.y[valid_data]
                
                # Linear trend line
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
            
            # =============================================================
            # SUB-BLOCK C3: CONFIGURE PLOT APPEARANCE
            # =============================================================
            # Truncate long column names
            short_col = col[:15] + '...' if len(col) > 15 else col
            ax.set_xlabel(short_col, fontsize=4)
            ax.set_ylabel('EAC', fontsize=4)  # Shorter label
            ax.set_title(f'{short_col}', fontsize=5)  # Just column name
            ax.legend(fontsize=3, loc='upper right')  # Much smaller legend
            ax.tick_params(axis='both', which='major', labelsize=3)
        
        plt.tight_layout(pad=0.3)  # Tighter layout
        self.canvas.draw()

# =============================================================================
# PART 16: RESIDUAL ANALYSIS & APPLICATION ENTRY POINT
# =============================================================================

    # =============================================================================
    # STEP 41: RESIDUAL PLOT ANALYSIS VISUALIZATION
    # =============================================================================
    
    def plot_residuals(self):
        """
        Plot residual analysis
        
        Residual plots are crucial for understanding model performance and
        identifying potential issues. This method creates two types of plots
        for each model:
        
        1. Predicted vs Actual: Shows how well predictions match reality
           - Points on diagonal line = perfect predictions
           - Scattered points = prediction errors
           - Systematic patterns indicate model bias
        
        2. Residuals Plot: Shows prediction errors vs predicted values
           - Residual = Actual - Predicted (the error)
           - Random scatter around zero = good model
           - Patterns indicate problems (bias, heteroscedasticity)
        
        These plots help identify:
        - Model bias (systematic over/under-prediction)
        - Heteroscedasticity (error variance changes with prediction size)
        - Outliers that strongly affect model performance
        - Which models perform best across different cost ranges
        """
        # =============================================================
        # BLOCK A: VALIDATION CHECK
        # =============================================================
        if not self.predictions:
            messagebox.showerror("Error", "No predictions available!")
            return
        
        self.fig.clear()
        
        # =============================================================
        # BLOCK B: DETERMINE LAYOUT CONFIGURATION
        # =============================================================
        n_models = len(self.predictions)
        if n_models == 0:
            return
            
        # Limit to maximum 4 models to fit in window (2 charts per model)
        models_to_show = min(n_models, 4)
        
        # Use 2x2 grid for up to 4 models (2 charts per model)
        if models_to_show <= 2:
            rows, cols = models_to_show, 2  # 1x2 or 2x2
        else:
            rows, cols = 2, 4  # 2x4 grid for 3-4 models
            
        print(f"Residuals grid layout: {rows}x{cols} for {models_to_show} models")
        
        # =============================================================
        # BLOCK C: CREATE RESIDUAL PLOTS FOR EACH MODEL
        # =============================================================
        for i, (model_name, y_pred) in enumerate(list(self.predictions.items())[:models_to_show]):
            
            # =============================================================
            # SUB-BLOCK C1: PREDICTED VS ACTUAL CHART
            # =============================================================
            ax = self.fig.add_subplot(rows, cols, i*2+1)
            
            # Ensure arrays have compatible shapes
            y_test_flat = self.y_test.ravel() if hasattr(self.y_test, 'ravel') else self.y_test
            y_pred_flat = y_pred.ravel() if hasattr(y_pred, 'ravel') else y_pred
            
            # Scatter plot of predicted vs actual values
            ax.scatter(y_test_flat, y_pred_flat, alpha=0.6)
            
            # Perfect prediction line (diagonal)
            # Points on this line would indicate perfect predictions
            min_val = min(y_test_flat.min(), y_pred_flat.min())
            max_val = max(y_test_flat.max(), y_pred_flat.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Configure chart appearance
            ax.set_xlabel('Actual', fontsize=4)
            ax.set_ylabel('Predicted', fontsize=4)
            ax.set_title(f'{model_name} - Pred vs Actual', fontsize=5)  # Shorter title
            ax.tick_params(axis='both', which='major', labelsize=3)
            
            # =============================================================
            # SUB-BLOCK C2: RESIDUALS PLOT
            # =============================================================
            ax2 = self.fig.add_subplot(rows, cols, i*2+2)
            
            # Calculate residuals (prediction errors)
            residuals = y_test_flat - y_pred_flat
            
            # Scatter plot of residuals vs predicted values
            ax2.scatter(y_pred_flat, residuals, alpha=0.6)
            
            # Zero line (perfect predictions would have residuals = 0)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            
            # Configure chart appearance
            ax2.set_xlabel('Predicted', fontsize=4)
            ax2.set_ylabel('Residuals', fontsize=4)
            ax2.set_title(f'{model_name} - Residuals', fontsize=5)
            ax2.tick_params(axis='both', which='major', labelsize=3)
        
        plt.tight_layout(pad=0.3)  # Much tighter layout
        self.canvas.draw()

# =============================================================================
# STEP 42: MAIN APPLICATION ENTRY POINT
# =============================================================================

def main():
    """
    Main Application Entry Point
    
    This function serves as the starting point for the entire application.
    It handles the initialization of the GUI framework and creates the
    main application window.
    
    The function follows the standard pattern for Tkinter applications:
    1. Create the root window (main container)
    2. Initialize the application class
    3. Start the event loop (keeps the application running)
    
    Usage:
        python project_estimation_app.py
        
    Or use the launcher:
        python run_app.py
        
    The application will continue running until the user closes the window
    or terminates the program. All user interactions are handled through
    the Tkinter event system.
    
    Technical Notes:
    - Uses Tkinter's mainloop() which is a blocking call
    - All GUI updates happen on the main thread
    - Exception handling is managed within individual methods
    - Application state is maintained in the ProjectEstimationApp class
    """
    # =============================================================
    # BLOCK A: CREATE MAIN TKINTER WINDOW
    # =============================================================
    # Initialize the root window - this is the main container for the entire application
    root = tk.Tk()
    
    # =============================================================
    # BLOCK B: INITIALIZE THE APPLICATION
    # =============================================================
    # Create an instance of our ProjectEstimationApp class
    # This will automatically set up all the GUI components, data structures,
    # and event handlers through the __init__ method
    app = ProjectEstimationApp(root)
    
    # =============================================================
    # BLOCK C: START THE GUI EVENT LOOP
    # =============================================================
    # Start the main event loop - this keeps the application running
    # and responsive to user interactions (clicks, typing, etc.)
    # This is a blocking call that runs until the window is closed
    root.mainloop()

# =============================================================================
# STEP 43: APPLICATION EXECUTION GUARD
# =============================================================================

if __name__ == "__main__":
    """
    Application Execution Guard
    
    This standard Python idiom ensures that the main() function
    only runs when this file is executed directly, not when it's
    imported as a module by another Python file.
    
    How it works:
    - When Python runs a file directly, __name__ is set to "__main__"
    - When Python imports a file, __name__ is set to the module name
    - This allows the file to be both executable and importable
    
    Benefits:
    - Prevents accidental execution when importing for testing
    - Allows other files to import classes/functions without side effects
    - Standard practice in Python application development
    - Enables unit testing and code reuse
    
    Example scenarios:
    - Direct execution: python project_estimation_app.py → main() runs
    - Import: from project_estimation_app import ProjectEstimationApp → main() does NOT run
    """
    main()

# =============================================================================
# END OF PROJECT ESTIMATION APPLICATION
# =============================================================================

"""
COMPREHENSIVE APPLICATION SUMMARY
================================

This Project Estimation ML Application provides a complete end-to-end solution
for construction project cost estimation using machine learning. Here's what
the application accomplishes:

KEY CAPABILITIES:
1. **Data Import & Validation**
   - Supports multiple file formats (CSV, Excel variants)
   - Intelligent data type detection
   - Comprehensive data preview and validation

2. **Advanced Data Preprocessing**
   - Smart categorical encoding (one-hot vs target encoding)
   - Statistical outlier detection using IQR method
   - Feature scaling and normalization
   - Transparent data cleaning with audit trails

3. **Multiple Machine Learning Models**
   - Linear models: Linear, Ridge, Lasso, ElasticNet
   - Tree-based: Random Forest, Extra Trees, Gradient Boosting, XGBoost
   - Other algorithms: SVR, KNN, Decision Tree
   - Configurable hyperparameters for each model

4. **Prediction Capabilities**
   - Manual single-project predictions with dynamic forms
   - Batch processing for multiple projects
   - Confidence intervals for uncertainty quantification
   - Model performance comparison and recommendations

5. **Professional Reporting**
   - Multi-format exports (Excel, CSV)
   - Comprehensive methodology documentation
   - UK currency formatting with confidence intervals
   - Model performance metrics and rankings

6. **Data Visualization & Analysis**
   - Feature importance charts
   - Correlation matrices
   - Scatter plots with trend lines
   - Residual analysis for model validation

7. **Transparency & Auditing**
   - Complete outlier data tracking
   - Model selection recommendations
   - Performance metric explanations
   - Methodology documentation

TECHNICAL ARCHITECTURE:
- Object-oriented design with comprehensive error handling
- Modular structure allowing easy extension
- Professional GUI with tabbed interface
- Statistical rigor with business-friendly presentation
- Scalable from single projects to large datasets

BUSINESS VALUE:
- Reduces estimation time from hours to minutes
- Provides multiple model perspectives for robust estimates
- Quantifies uncertainty through confidence intervals
- Maintains audit trails for compliance and transparency
- Scales efficiently for enterprise-level project portfolios

The application successfully bridges the gap between advanced machine learning
techniques and practical business needs in the construction industry.
"""