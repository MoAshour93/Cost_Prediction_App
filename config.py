# ==============================================================================
# CONFIGURATION MODULE
# ==============================================================================
#
# PURPOSE: Central configuration hub for the Project Estimation ML Application
# 
# WHAT THIS FILE DOES:
# - Stores ALL application settings in one place
# - Makes the app easy to customize without touching code
# - Provides consistent defaults across the entire application
# - Centralizes error messages and validation rules
#
# BUSINESS BENEFITS:
# - Easy to modify settings without being a programmer
# - Consistent behavior across all parts of the application
# - Quick customization for different business needs
# - Reduces errors by having settings in one location
#
# REAL-WORLD ANALOGY:
# Like a master control panel for a factory - all the important settings,
# limits, and preferences are in one place where managers can easily
# adjust them without needing to understand the machinery
# ==============================================================================

# STEP 1: IMPORT REQUIRED LIBRARIES
import os                    # Operating system interface for file operations
from datetime import datetime  # Date and time handling for timestamps


# ==============================================================================
# MAIN CONFIGURATION CLASS
# ==============================================================================
class Config:
    """
    MASTER SETTINGS CLASS: Contains all application configuration settings
    
    WHY WE USE A CLASS FOR CONFIGURATION:
    - Groups related settings together logically
    - Easy to access from anywhere in the application
    - Can include helper methods for validation
    - Prevents accidentally changing settings during runtime
    
    HOW TO USE:
    - Import this class in other files: from config import Config
    - Access settings like: Config.APP_NAME or Config.CHUNK_SIZE
    - Call methods like: Config.validate_file_size(filepath)
    """
    
    # --------------------------------------------------------------------------
    # APPLICATION IDENTITY AND DISPLAY SETTINGS
    # --------------------------------------------------------------------------
    # These settings control the basic application information and window appearance
    
    APP_NAME = "Project Estimation ML App"  # Application title shown in window
    VERSION = "1.2.3"                       # Current version for tracking updates
    APP_WIDTH = 1200                        # Default window width in pixels
    APP_HEIGHT = 800                        # Default window height in pixels
    
    # BUSINESS PURPOSE: Professional appearance and consistent branding
    
    # --------------------------------------------------------------------------
    # DATA PROCESSING CONFIGURATION
    # --------------------------------------------------------------------------
    # Controls how data is split and processed for machine learning
    
    DEFAULT_TRAIN_SIZE = 0.8    # Use 80% of data for training (industry standard)
    MIN_TRAIN_SIZE = 0.5        # Minimum 50% for training (ensures enough learning data)
    MAX_TRAIN_SIZE = 0.95       # Maximum 95% for training (ensures enough test data)
    NUMERIC_THRESHOLD = 0.7     # Column is numeric if 70% of values are numbers
    
    # BUSINESS EXPLANATION:
    # - Training data teaches the AI model
    # - Test data validates how well the model learned
    # - 80/20 split is proven to work well in most business scenarios
    
    # --------------------------------------------------------------------------
    # OUTLIER DETECTION SETTINGS
    # --------------------------------------------------------------------------
    # Controls automatic detection and removal of unusual data points
    
    IQR_MULTIPLIER = 1.5           # Standard statistical multiplier for outlier detection
    MAX_OUTLIER_PERCENTAGE = 0.2   # Don't remove more than 20% of data as outliers
    
    # BUSINESS PURPOSE:
    # - Outliers can skew predictions (like including Bill Gates in average income)
    # - Remove obvious errors, but keep enough data for reliable predictions
    # - 20% limit prevents over-cleaning of legitimate unusual cases
    
    # --------------------------------------------------------------------------
    # AI MODEL DEFAULT PARAMETERS
    # --------------------------------------------------------------------------
    # Pre-configured settings for each type of machine learning model
    
    MODEL_DEFAULTS = {
        # RANDOM FOREST: Like asking multiple experts and averaging their opinions
        'RandomForest': {
            'n_estimators': 100,        # Number of "expert opinions" (decision trees)
            'max_depth': 10,            # How deep each expert can think
            'min_samples_split': 2,     # Minimum data needed to make a decision
            'min_samples_leaf': 1,      # Minimum data in final decision bucket
            'random_state': 42          # Ensures reproducible results
        },
        
        # XGBOOST: Advanced algorithm that learns from previous mistakes
        'XGBoost': {
            'n_estimators': 100,           # Number of learning iterations
            'learning_rate': 0.1,          # How fast the model learns (0.1 = moderate)
            'max_depth': 6,                # Complexity limit to prevent overthinking
            'random_state': 42,            # Ensures reproducible results
            'objective': 'reg:squarederror' # Optimizes for prediction accuracy
        },
        
        # LINEAR REGRESSION: Simple straight-line relationships
        'LinearRegression': {
            'fit_intercept': True,      # Allow the line to not pass through zero
            'normalize': False          # Don't automatically scale the data
        }
    }
    
    # BUSINESS EXPLANATION:
    # - These are proven default settings that work well for most business cases
    # - Random Forest: Good for complex patterns, less prone to errors
    # - XGBoost: Often wins competitions, great for structured business data
    # - Linear Regression: Simple, interpretable, good for basic relationships
    
    # --------------------------------------------------------------------------
    # VISUALIZATION AND REPORTING SETTINGS
    # --------------------------------------------------------------------------
    # Controls the appearance and format of charts and reports
    
    PLOT_STYLE = 'seaborn-v0_8'     # Professional-looking chart style
    FIGURE_SIZE = (12, 8)           # Chart dimensions in inches
    DPI = 100                       # Chart resolution (dots per inch)
    COLOR_PALETTE = 'viridis'       # Color scheme (colorblind-friendly)
    MAX_FEATURES_PLOT = 10          # Show top 10 most important features only
    
    # BUSINESS PURPOSE:
    # - Professional charts for presentations to stakeholders
    # - Consistent visual branding across all reports
    # - Limit feature plots to top 10 for clarity (avoid information overload)
    
    # --------------------------------------------------------------------------
    # FILE HANDLING CONFIGURATION
    # --------------------------------------------------------------------------
    # Defines what types of files the application can work with
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']  # Excel and CSV files
    MAX_FILE_SIZE_MB = 500                         # Maximum 500MB file size
    DEFAULT_EXPORT_FORMAT = 'csv'                  # Save results as CSV by default
    
    # BUSINESS REASONING:
    # - CSV and Excel are the most common business data formats
    # - 500MB limit prevents system crashes from enormous files
    # - CSV default export ensures compatibility with other systems
    
    # --------------------------------------------------------------------------
    # LOGGING AND MONITORING SETTINGS
    # --------------------------------------------------------------------------
    # Controls how the application tracks its activities and errors
    
    LOG_LEVEL = 'INFO'                                                          # Level of detail in logs
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'       # Log message format
    LOG_FILE = f'project_estimation_{datetime.now().strftime("%Y%m%d")}.log'   # Daily log files
    
    # BUSINESS VALUE:
    # - Track application usage and performance
    # - Debug issues when they occur
    # - Audit trail for compliance and quality assurance
    # - Daily log files for easy organization
    
    # --------------------------------------------------------------------------
    # PERFORMANCE AND MEMORY MANAGEMENT
    # --------------------------------------------------------------------------
    # Settings that control application speed and memory usage
    
    MAX_DISPLAY_ROWS = 1000   # Don't show more than 1000 rows in tables (keeps UI responsive)
    CHUNK_SIZE = 5000         # Process large files in 5000-row chunks (prevents memory crashes)
    
    # BUSINESS BENEFITS:
    # - Keeps the application responsive with large datasets
    # - Prevents computer crashes from memory overload
    # - Users can still work with millions of rows efficiently
    
    # --------------------------------------------------------------------------
    # DATA VALIDATION RULES
    # --------------------------------------------------------------------------
    # Business rules that ensure data quality and prevent errors
    
    MIN_SAMPLES_FOR_TRAINING = 10   # Need at least 10 data points to train a model
    MIN_FEATURES = 1                # Need at least 1 input variable
    MAX_FEATURES = 1000             # Don't allow more than 1000 input variables
    
    # BUSINESS LOGIC:
    # - 10 samples minimum ensures statistical reliability
    # - 1000 features maximum prevents overly complex models
    # - These limits are based on machine learning best practices
    
    # --------------------------------------------------------------------------
    # USER-FRIENDLY ERROR MESSAGES
    # --------------------------------------------------------------------------
    # Clear, helpful error messages that non-technical users can understand
    
    ERROR_MESSAGES = {
        'no_file_loaded': "Please load a data file first!",
        'no_data_processed': "Please process the data first!",
        'no_models_trained': "Please train models first!",
        'insufficient_data': f"Need at least {MIN_SAMPLES_FOR_TRAINING} samples for training!",
        'file_too_large': f"File size exceeds {MAX_FILE_SIZE_MB}MB limit!",
        'unsupported_format': f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
        'target_not_numeric': "Target column must contain numeric values!",
        'no_numeric_features': "Dataset must contain at least one numeric feature!"
    }
    
    # BUSINESS PURPOSE:
    # - Help users understand what went wrong and how to fix it
    # - Reduce support calls by providing clear guidance
    # - Professional error handling improves user experience
    
    # --------------------------------------------------------------------------
    # POSITIVE FEEDBACK MESSAGES
    # --------------------------------------------------------------------------
    # Encouraging messages to confirm successful operations
    
    SUCCESS_MESSAGES = {
        'data_loaded': "Data loaded successfully!",
        'data_processed': "Data processing completed!",
        'models_trained': "All models trained successfully!",
        'predictions_saved': "Predictions exported successfully!",
        'outliers_removed': "Outliers detected and removed!"
    }
    
    # BUSINESS VALUE:
    # - Provide positive feedback to users
    # - Confirm that operations completed successfully
    # - Improve user confidence and experience

    # --------------------------------------------------------------------------
    # HELPER METHODS FOR CONFIGURATION MANAGEMENT
    # --------------------------------------------------------------------------
    
    @classmethod
    def get_model_params(cls, model_name):
        """
        HELPER METHOD: Retrieve default parameters for a specific AI model
        
        WHAT IT DOES:
        - Takes a model name (like 'RandomForest')
        - Returns the pre-configured settings for that model
        - Provides empty dictionary if model not found
        
        BUSINESS PURPOSE:
        - Ensures consistent model configuration across the application
        - Easy to modify default settings in one place
        - Prevents typos and configuration errors
        
        Parameters:
        - model_name (str): Name of the model ('RandomForest', 'XGBoost', etc.)
        
        Returns:
        - Dictionary of parameter settings for the specified model
        
        Example Usage:
        rf_params = Config.get_model_params('RandomForest')
        # Returns: {'n_estimators': 100, 'max_depth': 10, ...}
        """
        return cls.MODEL_DEFAULTS.get(model_name, {})
    
    @classmethod
    def validate_file_size(cls, file_path):
        """
        VALIDATION METHOD: Check if a file is within acceptable size limits
        
        WHAT IT DOES:
        - Checks if the file exists
        - Calculates file size in megabytes
        - Compares against the maximum allowed size
        - Returns True if file is acceptable, False if too large
        
        BUSINESS PURPOSE:
        - Prevents system crashes from oversized files
        - Gives user immediate feedback about file size issues
        - Protects system resources and performance
        
        Parameters:
        - file_path (str): Path to the file to check
        
        Returns:
        - Boolean: True if file size is OK, False if too large or file doesn't exist
        
        Example Usage:
        if Config.validate_file_size('my_data.csv'):
            # File is OK to process
        else:
            # File is too large or doesn't exist
        """
        if os.path.exists(file_path):
            # Calculate file size in megabytes
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return size_mb <= cls.MAX_FILE_SIZE_MB
        return False
    
    @classmethod
    def get_supported_extensions(cls):
        """
        UTILITY METHOD: Get list of file formats the application can handle
        
        WHAT IT DOES:
        - Returns the list of supported file extensions
        - Useful for file dialog filters and validation
        - Centralizes the supported format information
        
        BUSINESS PURPOSE:
        - Consistent file format support across the application
        - Easy to add new formats by updating SUPPORTED_FORMATS list
        - Helpful for user interface and error messages
        
        Returns:
        - List of supported file extensions (e.g., ['.csv', '.xlsx', '.xls'])
        
        Example Usage:
        supported = Config.get_supported_extensions()
        # Returns: ['.csv', '.xlsx', '.xls']
        """
        return cls.SUPPORTED_FORMATS


# ==============================================================================
# CONFIGURATION USAGE EXAMPLES AND BEST PRACTICES
# ==============================================================================

"""
HOW TO USE THIS CONFIGURATION IN OTHER FILES:

1. IMPORTING THE CONFIG:
   from config import Config

2. ACCESSING BASIC SETTINGS:
   app_name = Config.APP_NAME
   chunk_size = Config.CHUNK_SIZE
   
3. GETTING MODEL PARAMETERS:
   rf_params = Config.get_model_params('RandomForest')
   
4. VALIDATING FILES:
   if Config.validate_file_size(file_path):
       # Process the file
   else:
       print(Config.ERROR_MESSAGES['file_too_large'])

5. USING ERROR MESSAGES:
   if not data_loaded:
       show_error(Config.ERROR_MESSAGES['no_file_loaded'])

CUSTOMIZATION EXAMPLES:

1. TO CHANGE CHUNK SIZE FOR BETTER PERFORMANCE:
   # Modify CHUNK_SIZE = 5000 to CHUNK_SIZE = 10000

2. TO ADD A NEW SUPPORTED FILE FORMAT:
   # Add '.json' to SUPPORTED_FORMATS list

3. TO ADJUST MODEL PARAMETERS:
   # Modify values in MODEL_DEFAULTS dictionary

4. TO CHANGE APPLICATION APPEARANCE:
   # Modify APP_WIDTH, APP_HEIGHT, PLOT_STYLE, etc.

BUSINESS BENEFITS OF THIS APPROACH:

✅ CENTRALIZED MANAGEMENT: All settings in one place
✅ EASY CUSTOMIZATION: No programming required for basic changes
✅ CONSISTENT BEHAVIOR: Same settings used everywhere
✅ ERROR PREVENTION: Validation methods prevent common mistakes
✅ PROFESSIONAL MESSAGING: Consistent user communication
✅ MAINTAINABILITY: Easy to update and modify over time
✅ SCALABILITY: Easy to add new settings as the app grows

RECOMMENDED WORKFLOW FOR CHANGES:

1. Identify what needs to be customized
2. Find the relevant setting in this file
3. Modify the value
4. Test the application to ensure it works
5. Document any changes for future reference
"""