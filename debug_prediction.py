#!/usr/bin/env python3
# ==============================================================================
# DEBUG PREDICTION MODULE
# ==============================================================================
#
# PURPOSE: Troubleshooting and testing system for machine learning predictions
#
# WHAT THIS MODULE DOES:
# - Tests the complete prediction pipeline without the user interface
# - Identifies where prediction failures occur in the process
# - Simulates real-world data scenarios to validate system robustness
# - Provides detailed diagnostic information for debugging
#
# BUSINESS BENEFITS:
# - Prevents prediction failures in production by testing offline
# - Reduces debugging time by isolating problems systematically
# - Validates that the AI system works with different data types
# - Ensures reliable predictions before deploying to users
# - Documents the complete prediction workflow for training
#
# WHEN TO USE THIS:
# - When predictions fail in the main application
# - To test new data formats or scenarios
# - For system validation before major releases
# - To understand how the prediction pipeline works
# - For training team members on the prediction process
#
# REAL-WORLD ANALOGY:
# Like a test kitchen in a restaurant where chefs perfect recipes
# before serving them to customers - catches problems early
# ==============================================================================

"""
Debug Prediction Issues
Testing prediction pipeline without GUI to identify root cause
"""

# STEP 1: IMPORT REQUIRED LIBRARIES
# These are the core tools needed for machine learning and data processing
import pandas as pd                                    # Data manipulation and analysis
import numpy as np                                     # Numerical computations and arrays
from sklearn.model_selection import train_test_split   # Split data for training/testing
from sklearn.preprocessing import StandardScaler       # Scale features for some algorithms
from sklearn.ensemble import RandomForestRegressor     # Random Forest machine learning model
from sklearn.linear_model import LinearRegression      # Linear regression model
import xgboost as xgb                                  # XGBoost advanced machine learning model


# ==============================================================================
# MAIN TESTING FUNCTION
# ==============================================================================
def test_prediction_pipeline():
    """
    COMPREHENSIVE PREDICTION PIPELINE TEST
    
    BUSINESS PURPOSE:
    This function simulates the entire prediction process from start to finish
    using realistic construction project data, helping identify where problems
    might occur and ensuring the system works reliably
    
    TESTING APPROACH:
    1. Create realistic sample data (like real construction projects)
    2. Process the data exactly as the main application would
    3. Train multiple AI models on this data
    4. Test predictions on both batch data and individual manual inputs
    5. Identify any failures and provide detailed diagnostic information
    
    WHAT GETS TESTED:
    - Data creation and structure
    - Data preprocessing (cleaning and preparation)
    - Model training with different algorithms
    - Batch predictions on test data
    - Manual input predictions (simulating user input)
    - Error handling and edge cases
    
    BUSINESS VALUE:
    - Ensures reliability before deploying to users
    - Identifies potential failure points
    - Validates the complete workflow
    - Provides confidence in system stability
    
    Returns:
    - Dictionary of prediction results from all tested models
    """
    
    print("=== TESTING PREDICTION PIPELINE ===")
    
    # --------------------------------------------------------------------------
    # STEP 1: CREATE REALISTIC SAMPLE DATA
    # --------------------------------------------------------------------------
    # This simulates real construction project data that the system might encounter
    
    # Set random seed for reproducible testing
    np.random.seed(42)
    n_samples = 200  # Create 200 sample construction projects
    
    # CONSTRUCTION PROJECT FEATURES
    # These represent the typical inputs for construction project estimation
    data = {
        # NUMERIC FEATURES (measurable quantities)
        'Project_Size': np.random.normal(1000, 300, n_samples),      # Square footage
        'Material_Cost': np.random.normal(50000, 15000, n_samples),  # Cost of materials
        'Labor_Hours': np.random.normal(2000, 500, n_samples),       # Estimated work hours
        
        # CATEGORICAL FEATURES (categories that affect cost)
        'Complexity': np.random.choice(['Low', 'Medium', 'High'], n_samples),        # Project complexity
        'Location': np.random.choice(['Urban', 'Rural', 'Suburban'], n_samples),     # Project location
        'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples), # Construction season
    }
    
    # --------------------------------------------------------------------------
    # STEP 2: CREATE REALISTIC TARGET VARIABLE
    # --------------------------------------------------------------------------
    # This simulates how different factors actually affect construction costs
    
    # BUSINESS LOGIC: How different factors affect final project cost
    complexity_multiplier = {'Low': 1.0, 'Medium': 1.2, 'High': 1.5}    # Complexity impact
    location_multiplier = {'Rural': 1.0, 'Suburban': 1.1, 'Urban': 1.3}  # Location impact  
    season_multiplier = {'Spring': 1.0, 'Summer': 1.05, 'Fall': 1.02, 'Winter': 1.1}  # Season impact
    
    # CALCULATE REALISTIC PROJECT COSTS
    eac = []  # Estimate at Completion (what we want to predict)
    for i in range(n_samples):
        # Base cost calculation (materials + labor)
        base_cost = data['Material_Cost'][i] + (data['Labor_Hours'][i] * 50)  # $50/hour labor rate
        
        # Apply business multipliers
        complexity_mult = complexity_multiplier[data['Complexity'][i]]
        location_mult = location_multiplier[data['Location'][i]]
        season_mult = season_multiplier[data['Season'][i]]
        
        # Add realistic randomness (unexpected costs)
        noise = np.random.normal(0, 5000)
        final_cost = base_cost * complexity_mult * location_mult * season_mult + noise
        
        # Ensure positive values (can't have negative project costs)
        eac.append(max(final_cost, 10000))
    
    # Add target variable to dataset
    data['Estimate_at_Completion'] = eac
    df = pd.DataFrame(data)
    
    # DISPLAY SAMPLE DATA INFORMATION
    print(f"Created sample dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:")
    print(df.head())
    
    # --------------------------------------------------------------------------
    # STEP 3: SIMULATE DATA PROCESSING PIPELINE
    # --------------------------------------------------------------------------
    # This replicates the exact data preparation steps from the main application
    
    print("\\n=== SIMULATING DATA PROCESSING ===")
    
    # STEP 3A: Feature Type Detection
    # Identify which columns are numeric vs. categorical (text)
    numeric_cols = ['Project_Size', 'Material_Cost', 'Labor_Hours']
    text_cols = ['Complexity', 'Location', 'Season']
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Text columns: {text_cols}")
    
    # STEP 3B: Categorical Encoding Preparation
    # Machine learning algorithms need numbers, so we convert text to numbers
    processed_df = df.copy()
    feature_names = numeric_cols.copy()         # Track final feature names
    categorical_encoding_method = {}            # Track how each column was encoded
    label_encoders = {}                        # Store encoding mappings
    
    # PROCESS EACH CATEGORICAL COLUMN
    for col in text_cols:
        unique_count = df[col].nunique()
        print(f"Column {col} has {unique_count} unique values")
        
        # ENCODING STRATEGY DECISION
        if unique_count <= 3:  # Low cardinality: use one-hot encoding
            print(f"Applying one-hot encoding to {col}")
            
            # ONE-HOT ENCODING: Create binary columns for each category
            # Example: 'High', 'Medium', 'Low' becomes 'Complexity_High', 'Complexity_Medium'
            dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
            processed_df = pd.concat([processed_df, dummies], axis=1)
            processed_df.drop(columns=[col], inplace=True)
            feature_names.extend(list(dummies.columns))
            categorical_encoding_method[col] = 'onehot'
            
        else:  # High cardinality: use target encoding
            print(f"Applying target encoding to {col}")
            
            # TARGET ENCODING: Replace categories with average target value
            # Example: 'Urban' becomes the average project cost for urban projects
            target_means = processed_df.groupby(col)['Estimate_at_Completion'].mean()
            global_mean = processed_df['Estimate_at_Completion'].mean()
            
            # Store encoding information for later use
            label_encoders[col] = {
                'mapping': target_means.to_dict(),
                'global_mean': global_mean
            }
            processed_df[col] = processed_df[col].map(target_means).fillna(global_mean)
            feature_names.append(col)
            categorical_encoding_method[col] = 'target'
    
    print(f"Final feature names: {feature_names}")
    print(f"Processed DataFrame shape: {processed_df.shape}")
    print(f"Processed DataFrame columns: {list(processed_df.columns)}")
    
    # --------------------------------------------------------------------------
    # STEP 4: PREPARE DATA FOR MACHINE LEARNING
    # --------------------------------------------------------------------------
    
    # STEP 4A: Separate Features and Target
    X = processed_df.drop('Estimate_at_Completion', axis=1)  # Input features
    y = processed_df['Estimate_at_Completion']               # What we want to predict
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    # STEP 4B: Train/Test Split
    # Split data to train models and test their performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # STEP 4C: Feature Scaling
    # Some algorithms work better when all features are on similar scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # --------------------------------------------------------------------------
    # STEP 5: TRAIN MACHINE LEARNING MODELS
    # --------------------------------------------------------------------------
    # Train multiple types of models to compare performance and reliability
    
    print("\\n=== TRAINING MODELS ===")
    models = {}
    
    # MODEL 1: Linear Regression
    # Simple model that finds straight-line relationships
    # REQUIRES: Scaled features for optimal performance
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['Linear Regression'] = lr
    print("Trained Linear Regression")
    
    # MODEL 2: Random Forest
    # Advanced model that combines multiple decision trees
    # ADVANTAGE: Handles different feature scales naturally
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print("Trained Random Forest")
    
    # MODEL 3: XGBoost
    # State-of-the-art gradient boosting model
    # ADVANTAGE: Often provides best performance on structured data
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    print("Trained XGBoost")
    
    # --------------------------------------------------------------------------
    # STEP 6: TEST BATCH PREDICTIONS
    # --------------------------------------------------------------------------
    # Test predictions on the test set to ensure models work correctly
    
    print("\\n=== TEST SET PREDICTIONS ===")
    for model_name, model in models.items():
        try:
            # Use appropriate data format for each model
            if model_name == 'Linear Regression':
                pred = model.predict(X_test_scaled)  # Needs scaled features
            else:
                pred = model.predict(X_test)         # Uses original features
            print(f"{model_name}: {pred[:3]}... (first 3 predictions)")
        except Exception as e:
            print(f"{model_name} FAILED: {e}")
    
    # --------------------------------------------------------------------------
    # STEP 7: TEST MANUAL INPUT PREDICTION
    # --------------------------------------------------------------------------
    # This simulates a user entering new project data for prediction
    
    print("\\n=== MANUAL INPUT PREDICTION TEST ===")
    
    # SIMULATE USER INPUT
    # This represents what a user might enter in the application
    manual_input = {
        'Project_Size': '1200',      # User enters as text
        'Complexity': 'High',        # Dropdown selection
        'Location': 'Urban',         # Dropdown selection
        'Material_Cost': '65000',    # User enters as text
        'Labor_Hours': '2500',       # User enters as text
        'Season': 'Summer'           # Dropdown selection
    }
    
    print(f"Manual input: {manual_input}")
    
    # STEP 7A: Convert to DataFrame
    # Convert user input to the format our models expect
    input_df = pd.DataFrame([manual_input])
    print(f"Input DataFrame: {input_df}")
    
    # STEP 7B: Data Type Conversion
    # Convert text numbers to actual numbers
    processed_input = input_df.copy()
    
    for col in numeric_cols:
        if col in processed_input.columns:
            processed_input[col] = pd.to_numeric(processed_input[col], errors='coerce')
    
    print(f"After numeric conversion: {processed_input}")
    
    # STEP 7C: Apply Categorical Encoding
    # Convert categorical values using the same methods as training data
    for col in text_cols:
        if col in processed_input.columns:
            if col in categorical_encoding_method:
                method = categorical_encoding_method[col]
                print(f"Encoding {col} using {method}")
                
                if method == 'onehot':
                    # ONE-HOT ENCODING for manual input
                    # Create binary columns matching training data structure
                    dummies = pd.get_dummies(processed_input[col], prefix=col, drop_first=True)
                    print(f"Generated dummies: {list(dummies.columns)}")
                    
                    # Align with training columns (ensure same structure)
                    training_dummy_cols = [f for f in feature_names if f.startswith(f"{col}_")]
                    print(f"Expected dummy columns: {training_dummy_cols}")
                    
                    # Add missing columns (categories not present in this input)
                    for dummy_col in training_dummy_cols:
                        if dummy_col not in dummies.columns:
                            dummies[dummy_col] = 0
                    
                    # Remove extra columns and reorder to match training
                    dummies = dummies.reindex(columns=training_dummy_cols, fill_value=0)
                    
                    # Add encoded columns to processed data
                    processed_input = pd.concat([processed_input, dummies], axis=1)
                    processed_input.drop(columns=[col], inplace=True)
                    
                elif method == 'target':
                    # TARGET ENCODING for manual input
                    # Use the same mapping created during training
                    encoder_info = label_encoders[col]
                    mapping = encoder_info['mapping']
                    global_mean = encoder_info['global_mean']
                    
                    original_value = processed_input[col].iloc[0]
                    processed_input[col] = processed_input[col].map(mapping).fillna(global_mean)
                    encoded_value = processed_input[col].iloc[0]
                    print(f"Target encoded {col}: {original_value} -> {encoded_value}")
    
    print(f"After categorical encoding: {processed_input}")
    print(f"Processed input shape: {processed_input.shape}")
    print(f"Processed input columns: {list(processed_input.columns)}")
    
    # STEP 7D: Feature Alignment
    # Ensure input has exactly the same features as training data
    print(f"\\nAligning with training features: {feature_names}")
    processed_input = processed_input.reindex(columns=feature_names, fill_value=0)
    print(f"After alignment - shape: {processed_input.shape}")
    print(f"After alignment - columns: {list(processed_input.columns)}")
    
    # --------------------------------------------------------------------------
    # STEP 8: MAKE PREDICTIONS ON MANUAL INPUT
    # --------------------------------------------------------------------------
    # Test each model's ability to predict on the manually entered data
    
    print("\\n=== MAKING PREDICTIONS ===")
    predictions = {}
    
    for model_name, model in models.items():
        try:
            print(f"Predicting with {model_name}...")
            
            if model_name == 'Linear Regression':
                # Linear Regression requires scaled features
                print(f"Using scaled features - input shape: {processed_input.shape}")
                input_scaled = scaler.transform(processed_input)
                print(f"Scaled input shape: {input_scaled.shape}")
                pred = model.predict(input_scaled)[0]
            else:
                # Other models use original features
                print(f"Using original features - input shape: {processed_input.shape}")
                pred = model.predict(processed_input)[0]
            
            predictions[model_name] = pred
            print(f"✅ {model_name}: £{pred:,.2f}")
            
        except Exception as e:
            # Capture and display detailed error information
            print(f"❌ {model_name} FAILED: {e}")
            import traceback
            traceback.print_exc()  # Print full error traceback for debugging
            predictions[model_name] = f"Error: {e}"
    
    print(f"\\nFinal predictions: {predictions}")
    
    # --------------------------------------------------------------------------
    # STEP 9: GENERATE SUMMARY REPORT
    # --------------------------------------------------------------------------
    # Provide clear summary of testing results for business stakeholders
    
    successful = sum(1 for p in predictions.values() if not isinstance(p, str))
    total = len(predictions)
    print(f"\\n=== SUMMARY ===")
    print(f"Successful predictions: {successful}/{total}")
    
    if successful > 0:
        # Calculate statistics for successful predictions
        valid_preds = [p for p in predictions.values() if not isinstance(p, str)]
        print(f"Average prediction: £{np.mean(valid_preds):,.2f}")
        print(f"Prediction range: £{np.min(valid_preds):,.2f} - £{np.max(valid_preds):,.2f}")
    
    return predictions


# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    """
    MAIN EXECUTION: Run the complete prediction pipeline test
    
    This section executes when the script is run directly (not imported)
    
    BUSINESS USAGE:
    - Run this script whenever prediction issues occur
    - Use for system validation before releases
    - Execute during development to test changes
    - Run as part of automated testing suites
    
    EXPECTED OUTPUT:
    - Detailed step-by-step processing information
    - Success/failure status for each prediction model
    - Specific error messages if problems occur
    - Summary statistics for successful predictions
    
    TROUBLESHOOTING:
    If this test fails, it indicates problems in:
    - Data processing logic
    - Model training procedures
    - Feature encoding methods
    - Prediction pipeline integration
    
    HOW TO USE RESULTS:
    - Successful test = prediction system is working correctly
    - Failed test = specific error messages guide debugging
    - Partial success = identifies which models have issues
    """
    
    print("=" * 60)
    print("PREDICTION PIPELINE DEBUG SESSION")
    print("=" * 60)
    print("This test validates the complete prediction workflow")
    print("from data processing through final predictions.\\n")
    
    # Execute the comprehensive test
    predictions = test_prediction_pipeline()
    
    print("\\n" + "=" * 60)
    print("DEBUG SESSION COMPLETED")
    print("=" * 60)
    print("Review the output above for any errors or issues.")
    print("All successful predictions indicate a healthy system.")
    print("Any failures provide specific debugging information.")


# ==============================================================================
# DEBUGGING GUIDE FOR DEVELOPERS
# ==============================================================================
"""
COMMON ISSUES AND SOLUTIONS:

1. IMPORT ERRORS:
   Problem: Missing required libraries
   Solution: Install missing packages with pip install

2. SHAPE MISMATCH ERRORS:
   Problem: Input features don't match training features
   Solution: Check feature alignment and encoding consistency

3. PREDICTION ERRORS:
   Problem: Model can't process input format
   Solution: Verify data types and scaling requirements

4. ENCODING ERRORS:
   Problem: New categories not seen in training
   Solution: Check fallback handling (global_mean usage)

5. MEMORY ERRORS:
   Problem: Large datasets or models
   Solution: Reduce sample size or model complexity

PERFORMANCE BENCHMARKS:
- All models should complete training in < 5 seconds
- Predictions should complete in < 1 second
- Memory usage should stay under 1GB for test data

EXPECTED RESULTS:
- Linear Regression: Simple baseline predictions
- Random Forest: More sophisticated predictions
- XGBoost: Often highest accuracy predictions
- All predictions should be positive numbers
- Prediction range should be realistic for construction costs

WHEN TO RUN THIS TEST:
- Before deploying new versions
- When prediction accuracy seems low
- After changing data processing logic
- When adding new features or models
- During troubleshooting user-reported issues
"""