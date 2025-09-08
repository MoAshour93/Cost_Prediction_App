# ==============================================================================
# DATA VALIDATOR MODULE
# ==============================================================================
#
# PURPOSE: Quality control system for incoming business data
#
# WHAT THIS MODULE DOES:
# - Acts as a "data quality inspector" before machine learning begins
# - Identifies problems that could cause poor predictions or system crashes
# - Provides actionable recommendations to improve data quality
# - Ensures data meets minimum standards for reliable AI modeling
#
# BUSINESS BENEFITS:
# - Prevents "garbage in, garbage out" scenarios
# - Saves time by catching data issues early
# - Improves prediction accuracy by ensuring quality input
# - Provides clear guidance on how to fix data problems
# - Reduces failed modeling attempts and debugging time
#
# REAL-WORLD ANALOGY:
# Like a quality control inspector in a manufacturing plant who checks
# raw materials before they go into production - catches defects early
# before they become expensive problems down the line
# ==============================================================================

# STEP 1: IMPORT NECESSARY LIBRARIES
import pandas as pd                                    # Data manipulation and analysis
import numpy as np                                     # Numerical computations
from typing import Tuple, Dict, List, Optional        # Type hints for code clarity
import warnings                                        # Handle warning messages
from config import Config                              # Application configuration settings


# ==============================================================================
# MAIN DATA VALIDATOR CLASS
# ==============================================================================
class DataValidator:
    """
    COMPREHENSIVE DATA QUALITY INSPECTOR
    
    WHAT IT DOES:
    - Examines every aspect of your business data
    - Identifies potential problems before they cause issues
    - Provides specific recommendations for improvement
    - Generates detailed quality reports for stakeholders
    
    BUSINESS VALUE:
    - Prevents failed AI projects due to poor data quality
    - Saves money by catching problems early
    - Improves prediction accuracy through better data
    - Provides audit trail for data quality compliance
    
    HOW IT WORKS:
    1. Loads and examines the dataset structure
    2. Checks for missing values, duplicates, and inconsistencies
    3. Validates that target variable is suitable for prediction
    4. Analyzes feature quality and usefulness
    5. Ensures adequate sample size for reliable modeling
    6. Generates actionable recommendations
    """
    
    def __init__(self):
        """
        INITIALIZATION: Set up the validation tracking system
        
        These variables track the results of all validation checks:
        - validation_results: Detailed findings and statistics
        - warnings_list: Issues that should be addressed but won't stop processing
        - errors_list: Critical problems that must be fixed before proceeding
        """
        self.validation_results = {}    # Store detailed validation findings
        self.warnings_list = []         # Non-critical issues (yellow flags)
        self.errors_list = []          # Critical problems (red flags)
    
    # --------------------------------------------------------------------------
    # MAIN VALIDATION ORCHESTRATOR
    # --------------------------------------------------------------------------
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """
        MASTER VALIDATION FUNCTION: Comprehensive data quality assessment
        
        BUSINESS PROCESS:
        This is like a complete quality audit of your data, checking everything
        that could affect the success of your AI project
        
        VALIDATION CHECKLIST:
        ✓ Dataset Structure - Is the data organized properly?
        ✓ Data Quality - Are there missing values, duplicates, or errors?
        ✓ Target Variable - Can we predict what we want to predict?
        ✓ Features - Do we have useful information to make predictions?
        ✓ Sample Size - Do we have enough data for reliable results?
        
        RETURN VALUE:
        Complete report with pass/fail status, specific issues found,
        and actionable recommendations for improvement
        
        Parameters:
        - df (DataFrame): The business data to validate
        
        Returns:
        - Dictionary containing comprehensive validation results
        """
        
        # RESET TRACKING VARIABLES for fresh validation
        self.validation_results = {}
        self.warnings_list = []
        self.errors_list = []
        
        # STEP 1: Basic Structure Validation
        # Check if data is organized correctly for machine learning
        self._validate_structure(df)
        
        # STEP 2: Data Quality Assessment
        # Look for missing values, duplicates, and data inconsistencies
        self._validate_data_quality(df)
        
        # STEP 3: Target Variable Validation
        # Ensure we can actually predict what the business wants to predict
        self._validate_target_variable(df)
        
        # STEP 4: Feature Analysis
        # Check if we have useful information to make predictions
        self._validate_features(df)
        
        # STEP 5: Sample Size Adequacy
        # Verify we have enough data for reliable machine learning
        self._validate_sample_size(df)
        
        # STEP 6: Compile Final Results
        # Combine all findings into a comprehensive report
        self.validation_results.update({
            'is_valid': len(self.errors_list) == 0,        # Overall pass/fail status
            'warnings': self.warnings_list,                 # Issues to address
            'errors': self.errors_list,                     # Critical problems
            'recommendations': self._generate_recommendations()  # Action items
        })
        
        return self.validation_results
    
    # --------------------------------------------------------------------------
    # DATASET STRUCTURE VALIDATION
    # --------------------------------------------------------------------------
    def _validate_structure(self, df: pd.DataFrame):
        """
        STRUCTURAL INTEGRITY CHECK: Verify basic dataset organization
        
        WHAT WE'RE CHECKING:
        - Is there actually data in the file?
        - Are there enough columns to do machine learning?
        - Are column names unique and usable?
        
        BUSINESS IMPORTANCE:
        These are fundamental requirements - like checking if a building
        has a foundation before worrying about the paint color
        
        CHECKS PERFORMED:
        1. Empty dataset detection
        2. Minimum column count (need features + target)
        3. Duplicate column name detection
        4. Basic dataset information recording
        """
        
        # CHECK 1: Empty Dataset Detection
        if df.empty:
            self.errors_list.append("Dataset is empty")
            return  # Can't do other checks on empty data
        
        # CHECK 2: Minimum Column Requirement
        # Need at least 2 columns: 1 for input features, 1 for what we're predicting
        if len(df.columns) < 2:
            self.errors_list.append("Dataset must have at least 2 columns (features + target)")
        
        # CHECK 3: Duplicate Column Names
        # Each column must have a unique name for proper data handling
        if len(df.columns) != len(df.columns.unique()):
            self.errors_list.append("Dataset contains duplicate column names")
        
        # RECORD BASIC INFORMATION for reporting
        self.validation_results['shape'] = df.shape                    # Rows x Columns
        self.validation_results['columns'] = list(df.columns)          # Column names
        self.validation_results['dtypes'] = df.dtypes.to_dict()       # Data types
    
    # --------------------------------------------------------------------------
    # DATA QUALITY ASSESSMENT
    # --------------------------------------------------------------------------
    def _validate_data_quality(self, df: pd.DataFrame):
        """
        DATA QUALITY INSPECTION: Check for common data problems
        
        BUSINESS ANALOGY:
        Like inspecting products on an assembly line for defects
        - missing pieces, damaged items, duplicate items
        
        QUALITY CHECKS:
        1. Missing Values Analysis - Are there gaps in our data?
        2. Duplicate Records Detection - Do we have the same data twice?
        3. Data Type Consistency - Is numeric data stored properly?
        
        WHY THIS MATTERS:
        - Missing values can cause predictions to fail
        - Duplicates can skew results and waste processing time
        - Wrong data types prevent mathematical operations
        """
        
        # QUALITY CHECK 1: Missing Values Analysis
        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            missing_stats[col] = {
                'count': missing_count,
                'percentage': missing_percentage
            }
            
            # BUSINESS RULES for missing data tolerance
            if missing_percentage > 50:
                # Over 50% missing is problematic but may be workable
                self.warnings_list.append(f"Column '{col}' has {missing_percentage:.1f}% missing values")
            elif missing_percentage > 80:
                # Over 80% missing is usually unusable for predictions
                self.errors_list.append(f"Column '{col}' has {missing_percentage:.1f}% missing values (too high)")
        
        self.validation_results['missing_values'] = missing_stats
        
        # QUALITY CHECK 2: Duplicate Records Detection
        duplicate_count = df.duplicated().sum()
        self.validation_results['duplicate_rows'] = duplicate_count
        
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            # Warn if more than 10% of data is duplicated
            if duplicate_percentage > 10:
                self.warnings_list.append(f"Dataset contains {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)")
        
        # QUALITY CHECK 3: Data Type Consistency
        self._check_data_type_consistency(df)
    
    # --------------------------------------------------------------------------
    # DATA TYPE CONSISTENCY CHECKER
    # --------------------------------------------------------------------------
    def _check_data_type_consistency(self, df: pd.DataFrame):
        """
        DATA TYPE DETECTIVE: Find numbers disguised as text
        
        BUSINESS PROBLEM:
        Sometimes numeric data (like prices, quantities) gets stored as text
        This prevents mathematical operations and machine learning
        
        WHAT WE'RE LOOKING FOR:
        Columns that appear to be text but actually contain mostly numbers
        
        DETECTION LOGIC:
        1. Try to convert text columns to numbers
        2. If most values convert successfully, it's probably numeric data stored wrong
        3. Flag these for correction
        """
        inconsistencies = []
        
        for col in df.columns:
            # Only check columns that are currently stored as text
            if df[col].dtype == 'object':
                try:
                    # Attempt to convert to numeric, replacing failures with NaN
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    numeric_count = numeric_series.notna().sum()
                    total_non_null = df[col].notna().sum()
                    
                    # If most non-null values can be converted to numbers
                    if numeric_count > 0 and numeric_count / total_non_null > Config.NUMERIC_THRESHOLD:
                        inconsistencies.append(f"Column '{col}' appears to contain numeric data stored as text")
                except:
                    # If conversion fails completely, that's fine - probably truly text data
                    pass
        
        # Record and warn about inconsistencies found
        if inconsistencies:
            self.validation_results['type_inconsistencies'] = inconsistencies
            for inconsistency in inconsistencies:
                self.warnings_list.append(inconsistency)
    
    # --------------------------------------------------------------------------
    # TARGET VARIABLE VALIDATION
    # --------------------------------------------------------------------------
    def _validate_target_variable(self, df: pd.DataFrame):
        """
        TARGET VALIDATION: Ensure we can predict what the business wants
        
        BUSINESS IMPORTANCE:
        The target variable is what we're trying to predict (sales, costs, etc.)
        If this isn't suitable for machine learning, the entire project fails
        
        ASSUMPTION: Target variable is in the rightmost column
        (This is a common convention in machine learning datasets)
        
        VALIDATION CHECKLIST:
        ✓ Is the target variable numeric? (required for regression models)
        ✓ Does it have enough valid values?
        ✓ Is there sufficient variation in the values?
        ✓ Are the values distributed reasonably?
        
        STATISTICAL ANALYSIS:
        - Calculate mean, standard deviation, min, max, median
        - Check for skewness (heavily biased toward high or low values)
        - Detect zero variance (all values the same - can't predict)
        """
        
        # Skip validation if dataset is empty
        if df.empty:
            return
        
        # IDENTIFY TARGET COLUMN (assume rightmost column)
        target_col = df.columns[-1]
        target_series = df[target_col]
        
        # NUMERIC VALIDATION: Can we treat this as numbers for prediction?
        numeric_target = pd.to_numeric(target_series, errors='coerce')
        numeric_count = numeric_target.notna().sum()        # How many values converted successfully
        total_count = target_series.notna().sum()           # How many non-null values total
        
        # CHECK 1: Does target have any valid values?
        if total_count == 0:
            self.errors_list.append(f"Target column '{target_col}' has no valid values")
            return
        
        # CHECK 2: Is target sufficiently numeric?
        numeric_percentage = (numeric_count / total_count) * 100
        
        if numeric_percentage < 80:
            # Less than 80% numeric is problematic for regression
            self.errors_list.append(f"Target column '{target_col}' is not sufficiently numeric ({numeric_percentage:.1f}%)")
        elif numeric_percentage < 95:
            # 80-95% numeric is workable but worth noting
            self.warnings_list.append(f"Target column '{target_col}' has some non-numeric values ({100-numeric_percentage:.1f}%)")
        
        # STATISTICAL ANALYSIS of numeric target values
        if numeric_count > 0:
            target_stats = {
                'mean': float(numeric_target.mean()),         # Average value
                'std': float(numeric_target.std()),           # Spread of values
                'min': float(numeric_target.min()),           # Minimum value
                'max': float(numeric_target.max()),           # Maximum value
                'median': float(numeric_target.median()),     # Middle value
                'skewness': float(numeric_target.skew()) if numeric_target.std() > 0 else 0  # Distribution shape
            }
            
            self.validation_results['target_stats'] = target_stats
            
            # CHECK 3: Distribution Analysis
            # Highly skewed data can cause prediction problems
            if abs(target_stats['skewness']) > 2:
                self.warnings_list.append(f"Target variable is highly skewed (skewness: {target_stats['skewness']:.2f})")
            
            # CHECK 4: Variance Analysis
            # Zero variance means all values are identical - impossible to predict
            if target_stats['std'] == 0:
                self.errors_list.append("Target variable has zero variance (all values are identical)")
    
    # --------------------------------------------------------------------------
    # FEATURE VALIDATION
    # --------------------------------------------------------------------------
    def _validate_features(self, df: pd.DataFrame):
        """
        FEATURE QUALITY ASSESSMENT: Analyze input variables for prediction usefulness
        
        BUSINESS EXPLANATION:
        Features are the input variables we use to make predictions
        (like using sales history, market conditions to predict future sales)
        
        ASSUMPTION: All columns except the rightmost are features
        (Rightmost column is assumed to be the target variable)
        
        FEATURE ANALYSIS:
        1. Classify each feature as numeric or categorical
        2. Calculate statistics for numeric features
        3. Analyze categorical features for usefulness
        4. Check for features that won't help with prediction
        
        QUALITY INDICATORS:
        - Numeric features: mean, spread, uniqueness
        - Categorical features: number of categories, distribution
        - Zero variance features: same value everywhere (useless)
        - High cardinality: too many unique values (may cause problems)
        """
        
        # Skip if we don't have enough columns for features
        if len(df.columns) < 2:
            return
        
        # IDENTIFY FEATURE COLUMNS (all except rightmost which is target)
        feature_cols = df.columns[:-1]
        feature_analysis = {}
        
        # COUNTERS for summary statistics
        numeric_features = 0
        categorical_features = 0
        
        # ANALYZE EACH FEATURE INDIVIDUALLY
        for col in feature_cols:
            col_analysis = {}
            
            # STEP 1: Determine if feature is numeric or categorical
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            numeric_count = numeric_series.notna().sum()
            total_count = df[col].notna().sum()
            
            if total_count > 0:
                numeric_percentage = (numeric_count / total_count) * 100
                
                # NUMERIC FEATURE ANALYSIS
                if numeric_percentage > Config.NUMERIC_THRESHOLD * 100:
                    col_analysis['type'] = 'numeric'
                    numeric_features += 1
                    
                    # Calculate statistical summary for numeric features
                    if numeric_count > 0:
                        col_analysis['stats'] = {
                            'mean': float(numeric_series.mean()),              # Average value
                            'std': float(numeric_series.std()),               # Spread of values
                            'min': float(numeric_series.min()),               # Minimum value
                            'max': float(numeric_series.max()),               # Maximum value
                            'unique_values': int(numeric_series.nunique())    # How many different values
                        }
                        
                        # QUALITY CHECK: Zero variance detection
                        # If all values are the same, this feature won't help with prediction
                        if col_analysis['stats']['std'] == 0:
                            self.warnings_list.append(f"Feature '{col}' has zero variance")
                
                # CATEGORICAL FEATURE ANALYSIS
                else:
                    col_analysis['type'] = 'categorical'
                    categorical_features += 1
                    
                    # Analyze categorical distribution
                    unique_count = df[col].nunique()
                    col_analysis['unique_values'] = unique_count
                    col_analysis['value_counts'] = df[col].value_counts().head(10).to_dict()
                    
                    # QUALITY CHECK: High cardinality warning
                    # Too many categories can cause memory and performance issues
                    if unique_count > len(df) * 0.5:
                        self.warnings_list.append(f"Feature '{col}' has very high cardinality ({unique_count} unique values)")
            
            feature_analysis[col] = col_analysis
        
        # STORE RESULTS for reporting
        self.validation_results['feature_analysis'] = feature_analysis
        self.validation_results['numeric_features'] = numeric_features
        self.validation_results['categorical_features'] = categorical_features
        
        # FEATURE ADEQUACY CHECKS
        if numeric_features == 0:
            self.warnings_list.append("Dataset has no numeric features - model performance may be limited")
        
        if numeric_features + categorical_features == 0:
            self.errors_list.append("Dataset has no valid features for modeling")
    
    # --------------------------------------------------------------------------
    # SAMPLE SIZE VALIDATION
    # --------------------------------------------------------------------------
    def _validate_sample_size(self, df: pd.DataFrame):
        """
        SAMPLE SIZE ADEQUACY: Ensure sufficient data for reliable machine learning
        
        BUSINESS IMPORTANCE:
        Machine learning needs enough examples to learn patterns reliably
        Too little data = unreliable predictions and poor business decisions
        
        STATISTICAL RULES:
        1. Absolute minimum: 10 samples (from configuration)
        2. Samples-to-features ratio: at least 10:1 recommended
        3. More complex models need more data
        
        BUSINESS ANALOGY:
        Like training an employee - you need enough examples for them to
        learn the job properly. One or two examples aren't enough to
        understand complex business patterns.
        
        CHECKS PERFORMED:
        1. Minimum absolute sample count
        2. Samples-to-features ratio analysis
        3. Risk assessment for overfitting
        """
        
        # CALCULATE KEY METRICS
        n_samples = len(df)                          # Total number of data rows
        n_features = len(df.columns) - 1             # Number of input features (exclude target)
        
        # STORE METRICS for reporting
        self.validation_results['sample_size'] = n_samples
        self.validation_results['features_count'] = n_features
        
        # CHECK 1: Minimum Sample Size
        # Absolute minimum from business configuration
        if n_samples < Config.MIN_SAMPLES_FOR_TRAINING:
            self.errors_list.append(f"Insufficient samples: {n_samples} (minimum: {Config.MIN_SAMPLES_FOR_TRAINING})")
        
        # CHECK 2: Samples-to-Features Ratio
        # Rule of thumb: need at least 10 samples per feature for reliable learning
        if n_features > 0:
            ratio = n_samples / n_features
            self.validation_results['samples_per_feature'] = ratio
            
            # RATIO ASSESSMENT
            if ratio < 10:
                # Less than 10:1 is concerning but may be workable
                self.warnings_list.append(f"Low samples-to-features ratio: {ratio:.1f} (recommended: >10)")
            elif ratio < 5:
                # Less than 5:1 is high risk for overfitting (memorizing instead of learning)
                self.errors_list.append(f"Very low samples-to-features ratio: {ratio:.1f} (risk of overfitting)")
    
    # --------------------------------------------------------------------------
    # RECOMMENDATION ENGINE
    # --------------------------------------------------------------------------
    def _generate_recommendations(self) -> List[str]:
        """
        SMART RECOMMENDATIONS: Generate actionable advice based on validation findings
        
        BUSINESS PURPOSE:
        Don't just tell users what's wrong - tell them specifically how to fix it
        Provides clear action items for improving data quality
        
        RECOMMENDATION CATEGORIES:
        1. Data Quality Improvements
        2. Sample Size Enhancements
        3. Feature Engineering Suggestions
        4. Target Variable Transformations
        
        OUTPUT: Prioritized list of specific actions to take
        """
        recommendations = []
        
        # RECOMMENDATION 1: Missing Data Management
        if 'missing_values' in self.validation_results:
            # Identify columns with problematic missing data levels
            high_missing_cols = [col for col, stats in self.validation_results['missing_values'].items() 
                               if stats['percentage'] > 20]
            if high_missing_cols:
                recommendations.append(f"Consider removing or imputing columns with high missing values: {', '.join(high_missing_cols)}")
        
        # RECOMMENDATION 2: Sample Size Enhancement
        if self.validation_results.get('sample_size', 0) < 100:
            recommendations.append("Consider collecting more data for better model performance (recommended: >100 samples)")
        
        # RECOMMENDATION 3: Feature Enrichment
        if self.validation_results.get('numeric_features', 0) < 3:
            recommendations.append("Consider adding more numeric features for better predictive power")
        
        # RECOMMENDATION 4: Target Variable Transformation
        if 'target_stats' in self.validation_results:
            skewness = self.validation_results['target_stats']['skewness']
            if abs(skewness) > 1.5:
                recommendations.append("Consider log transformation for the target variable to reduce skewness")
        
        return recommendations
    
    # --------------------------------------------------------------------------
    # VALIDATION REPORT GENERATOR
    # --------------------------------------------------------------------------
    def print_validation_report(self):
        """
        COMPREHENSIVE REPORTING: Generate stakeholder-friendly validation report
        
        BUSINESS PURPOSE:
        - Provide clear, professional summary for non-technical stakeholders
        - Document data quality status for audit purposes
        - Give specific guidance on next steps
        
        REPORT SECTIONS:
        1. Dataset Overview (size, structure)
        2. Critical Errors (must fix before proceeding)
        3. Warnings (should address for best results)
        4. Recommendations (specific action items)
        5. Overall Status (pass/fail determination)
        """
        
        # REPORT HEADER
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        
        # DATASET OVERVIEW
        if 'shape' in self.validation_results:
            rows, cols = self.validation_results['shape']
            print(f"Dataset Shape: {rows:,} rows × {cols} columns")
        
        # CRITICAL ERRORS SECTION
        if self.errors_list:
            print("\n❌ CRITICAL ERRORS (Must Fix):")
            for error in self.errors_list:
                print(f"  • {error}")
        
        # WARNINGS SECTION
        if self.warnings_list:
            print("\n⚠️ WARNINGS (Should Address):")
            for warning in self.warnings_list:
                print(f"  • {warning}")
        
        # RECOMMENDATIONS SECTION
        if self.validation_results.get('recommendations'):
            print("\n💡 RECOMMENDATIONS (Action Items):")
            for rec in self.validation_results['recommendations']:
                print(f"  • {rec}")
        
        # OVERALL STATUS
        status = "✅ PASSED" if self.validation_results.get('is_valid') else "❌ FAILED"
        print(f"\nValidation Status: {status}")
        print("=" * 60)


# ==============================================================================
# STANDALONE VALIDATION FUNCTION
# ==============================================================================
def validate_data_file(file_path: str) -> Tuple[bool, Dict]:
    """
    QUICK FILE VALIDATION: One-function validation for external use
    
    BUSINESS USE CASE:
    Simple function that other parts of the application can call
    to quickly check if a data file is suitable for machine learning
    
    PROCESS:
    1. Load the data file (CSV or Excel)
    2. Run complete validation suite
    3. Return simple pass/fail + detailed results
    
    INTEGRATION POINT:
    This function can be called from user interfaces, batch processing
    systems, or automated quality checks
    
    Parameters:
    - file_path (str): Path to the data file to validate
    
    Returns:
    - Tuple of (is_valid: bool, validation_results: Dict)
      * is_valid: True if data passes all critical checks
      * validation_results: Detailed findings and recommendations
    """
    try:
        # STEP 1: Load the data file based on format
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return False, {'error': 'Unsupported file format'}
        
        # STEP 2: Run comprehensive validation
        validator = DataValidator()
        results = validator.validate_dataset(df)
        
        # STEP 3: Return results
        return results['is_valid'], results
        
    except Exception as e:
        # Handle file loading errors gracefully
        return False, {'error': f'Failed to load file: {str(e)}'}


# ==============================================================================
# DEMONSTRATION AND TESTING CODE
# ==============================================================================
if __name__ == "__main__":
    """
    DEMONSTRATION: Show how the data validator works with real examples
    
    This section runs when the file is executed directly (not imported)
    It demonstrates the validator's capabilities with sample data
    """
    
    print("=" * 60)
    print("DATA VALIDATOR DEMONSTRATION")
    print("=" * 60)
    
    # Create validator instance
    validator = DataValidator()
    
    # STEP 1: Create realistic sample business data
    print("Creating sample business dataset...")
    
    # Simulate a business dataset with common issues
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),               # Normal numeric feature
        'feature2': ['A', 'B', 'C'] * 33 + ['A'],            # Categorical feature
        'feature3': np.random.randint(1, 10, 100),            # Integer feature
        'target': np.random.normal(50, 10, 100)               # Target variable (what we predict)
    })
    
    # Add some realistic data quality issues
    # Add missing values to simulate real-world data
    sample_data.loc[0:5, 'feature1'] = np.nan
    sample_data.loc[10:12, 'feature2'] = None
    
    print(f"Sample dataset created: {sample_data.shape[0]} rows, {sample_data.shape[1]} columns")
    print(f"Features: {list(sample_data.columns[:-1])}")
    print(f"Target: {sample_data.columns[-1]}")
    
    # STEP 2: Run comprehensive validation
    print("\nRunning comprehensive data validation...")
    results = validator.validate_dataset(sample_data)
    
    # STEP 3: Display detailed report
    validator.print_validation_report()
    
    # STEP 4: Show programmatic access to results
    print("\n" + "=" * 40)
    print("PROGRAMMATIC RESULTS ACCESS")
    print("=" * 40)
    print(f"Overall validation status: {results['is_valid']}")
    print(f"Number of warnings: {len(results['warnings'])}")
    print(f"Number of errors: {len(results['errors'])}")
    print(f"Number of recommendations: {len(results['recommendations'])}")
    
    if 'target_stats' in results:
        target_stats = results['target_stats']
        print(f"Target variable mean: {target_stats['mean']:.2f}")
        print(f"Target variable std: {target_stats['std']:.2f}")
    
    print("\n✅ Demonstration completed!")
    print("This validator is ready to assess your business data quality.")