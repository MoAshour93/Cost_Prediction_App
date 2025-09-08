# =============================================================================
# MODEL EVALUATION AND COMPARISON UTILITIES
# =============================================================================
# This file contains tools to evaluate and compare machine learning models
# Think of it as a "report card generator" for AI models that tells us:
# - How accurate the model's predictions are
# - What types of errors it makes
# - How it compares to other models
# - What problems might exist and how to fix them
# =============================================================================

# STEP 1: IMPORT REQUIRED LIBRARIES
# =============================================================================
# These are like importing tools from a toolbox - each library provides
# specific functions we need for our analysis

import numpy as np                    # For mathematical calculations and arrays
import pandas as pd                   # For handling data tables (like Excel)
from sklearn.metrics import (         # Tools to measure model performance
    mean_squared_error,               # Measures average prediction errors (squared)
    mean_absolute_error,              # Measures average prediction errors (absolute)
    r2_score,                        # Measures how well model explains the data (0-1)
    mean_absolute_percentage_error    # Measures percentage errors
)
from sklearn.model_selection import cross_val_score  # For testing model reliability
from typing import Dict, List, Tuple, Optional       # For better code documentation
import matplotlib.pyplot as plt                      # For creating charts (not used here)
from scipy import stats                              # For statistical calculations
import warnings                                      # For handling warning messages


# STEP 2: MAIN MODEL EVALUATOR CLASS
# =============================================================================
# This is like creating a "Model Report Card Generator" - a tool that can
# evaluate any machine learning model and tell us how good it is
# =============================================================================

class ModelEvaluator:
    """
    COMPREHENSIVE MODEL EVALUATION AND COMPARISON TOOL
    
    Think of this as an automated teacher that grades machine learning models.
    It can:
    - Grade individual models (like giving a test score)
    - Compare multiple models (like ranking students in a class)
    - Identify problems (like finding weak areas in performance)
    - Generate detailed reports (like writing detailed feedback)
    """
    
    def __init__(self):
        """
        INITIALIZATION - Setting up the evaluator
        
        This creates two empty storage containers:
        - evaluation_results: Stores detailed grades for each model
        - model_rankings: Stores how models rank against each other
        """
        self.evaluation_results = {}    # Dictionary to store all model results
        self.model_rankings = {}        # Dictionary to store model rankings
    
    
    # STEP 3: MAIN EVALUATION METHOD
    # =============================================================================
    def evaluate_model(self, 
                      model_name: str,           # Name of the model (like "Student A")
                      y_true: np.ndarray,        # The correct answers
                      y_pred: np.ndarray,        # The model's predictions
                      model=None,                # The actual model (optional)
                      X_test: Optional[np.ndarray] = None) -> Dict:  # Test data (optional)
        """
        COMPREHENSIVE EVALUATION OF A SINGLE MODEL
        
        This is like giving a comprehensive exam to a student. We check:
        - Overall accuracy (how often they get it right)
        - Types of mistakes they make
        - Consistency of performance
        - Areas where they struggle
        
        INPUT EXPLANATION:
        - model_name: A label for this model (like "Random Forest Model")
        - y_true: The actual correct values we're trying to predict
        - y_pred: What the model predicted for those same cases
        - model: The trained model object (optional, for advanced metrics)
        - X_test: The input data used for predictions (optional)
        
        OUTPUT: A comprehensive report card with all performance metrics
        """
        
        results = {}  # This will store all our evaluation metrics
        
        # BLOCK 1: BASIC ACCURACY MEASUREMENTS
        # ====================================================================
        # These are the fundamental "report card grades" - how well did the
        # model perform overall?
        
        # Mean Squared Error (MSE) - penalizes big mistakes more heavily
        results['mse'] = mean_squared_error(y_true, y_pred)
        
        # Root Mean Squared Error (RMSE) - easier to interpret than MSE
        # This tells us the average size of prediction errors
        results['rmse'] = np.sqrt(results['mse'])
        
        # Mean Absolute Error (MAE) - average size of all errors
        # This treats all errors equally (doesn't penalize big mistakes extra)
        results['mae'] = mean_absolute_error(y_true, y_pred)
        
        # R² Score - how much of the variation in data does our model explain?
        # Scale: 0 (terrible) to 1 (perfect). Like a correlation strength measure
        results['r2'] = r2_score(y_true, y_pred)
        
        
        # BLOCK 2: PERCENTAGE ERROR CALCULATION
        # ====================================================================
        # Mean Absolute Percentage Error (MAPE) - errors as percentages
        # This is often easier for non-technical people to understand
        
        try:
            # Try the standard calculation
            results['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            # If that fails (due to zeros in data), use our safe method
            results['mape'] = self._safe_mape(y_true, y_pred)
        
        
        # BLOCK 3: ADVANCED METRICS (if we have the trained model)
        # ====================================================================
        # Adjusted R² - R² score that accounts for model complexity
        # More features = more complex model, so we adjust the score accordingly
        
        if model is not None and X_test is not None:
            n = len(y_true)                                    # Number of predictions
            p = X_test.shape[1] if hasattr(X_test, 'shape') else 1  # Number of features
            # Mathematical adjustment for complexity
            results['adj_r2'] = 1 - (1 - results['r2']) * (n - 1) / (n - p - 1)
        
        
        # BLOCK 4: RESIDUAL ANALYSIS
        # ====================================================================
        # Residuals = True Value - Predicted Value (the errors)
        # Analyzing errors helps us understand if the model has systematic problems
        
        residuals = y_true - y_pred  # Calculate all the prediction errors
        
        # Statistical properties of errors
        results['residual_std'] = np.std(residuals)      # How spread out are errors?
        results['residual_mean'] = np.mean(residuals)    # Are errors biased in one direction?
        results['residual_skewness'] = stats.skew(residuals)    # Are errors symmetric?
        results['residual_kurtosis'] = stats.kurtosis(residuals)  # How "peaked" is error distribution?
        
        
        # BLOCK 5: PRACTICAL ACCURACY MEASUREMENTS
        # ====================================================================
        # These tell us: "What percentage of predictions are within X% of the truth?"
        # This is very useful for business applications
        
        results['within_10_percent'] = self._accuracy_within_percentage(y_true, y_pred, 0.10)
        results['within_20_percent'] = self._accuracy_within_percentage(y_true, y_pred, 0.20)
        results['within_30_percent'] = self._accuracy_within_percentage(y_true, y_pred, 0.30)
        
        
        # BLOCK 6: OUTLIER DETECTION
        # ====================================================================
        # Identify predictions that are unusually far from the truth
        # This helps spot data quality issues or model problems
        
        results['outlier_predictions'] = self._count_outlier_predictions(y_true, y_pred)
        
        
        # BLOCK 7: PREDICTION CONFIDENCE INTERVALS
        # ====================================================================
        # Calculate confidence ranges around predictions
        # This tells us: "We're 95% confident the true value is in this range"
        
        results['prediction_bounds'] = self._calculate_prediction_bounds(y_pred, residuals)
        
        
        # BLOCK 8: STORE AND RETURN RESULTS
        # ====================================================================
        # Save results for this model and return the complete evaluation
        
        self.evaluation_results[model_name] = results  # Store for later comparison
        return results  # Return results immediately for use
    
    
    # STEP 4: HELPER METHODS FOR SPECIFIC CALCULATIONS
    # =============================================================================
    # These are specialized tools used by the main evaluation method
    # Think of them as specialized calculators for specific tasks
    # =============================================================================
    
    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        SAFE CALCULATION OF MEAN ABSOLUTE PERCENTAGE ERROR
        
        Problem: Regular MAPE calculation fails when true values contain zeros
        (because you can't divide by zero)
        
        Solution: This method handles zeros safely by excluding them from calculation
        
        Returns: MAPE as a percentage, or infinity if no valid calculations possible
        """
        # Find positions where true values are not zero
        non_zero_mask = y_true != 0
        
        # If ALL true values are zero, we can't calculate MAPE
        if np.sum(non_zero_mask) == 0:
            return np.inf  # Return infinity to indicate impossible calculation
        
        # Calculate MAPE only for non-zero true values
        percentage_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        return np.mean(percentage_errors) * 100  # Convert to percentage
    
    
    def _accuracy_within_percentage(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
        """
        CALCULATE PERCENTAGE OF PREDICTIONS WITHIN A THRESHOLD
        
        This answers: "What percentage of our predictions are within X% of the truth?"
        
        Example: If threshold = 0.20 (20%), this counts how many predictions
        are within 20% of the actual values
        
        INPUT:
        - y_true: Actual values
        - y_pred: Predicted values  
        - threshold: The percentage threshold (0.10 = 10%, 0.20 = 20%, etc.)
        
        OUTPUT: Percentage of predictions within the threshold
        """
        # Calculate percentage error for each prediction
        # Use maximum of absolute true value and small number to avoid division by zero
        percentage_errors = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))
        
        # Count how many errors are within threshold and convert to percentage
        return np.mean(percentage_errors <= threshold) * 100
    
    
    def _count_outlier_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        IDENTIFY OUTLIER PREDICTIONS USING STATISTICAL METHODS
        
        Outliers are predictions that are unusually far from the truth.
        We use the "Interquartile Range (IQR)" method to identify them.
        
        Method:
        1. Calculate residuals (errors)
        2. Find Q1 (25th percentile) and Q3 (75th percentile) of errors
        3. Calculate IQR = Q3 - Q1
        4. Anything beyond 1.5 * IQR from the median is considered an outlier
        
        OUTPUT: Dictionary with outlier count, percentage, and threshold used
        """
        residuals = y_true - y_pred  # Calculate prediction errors
        
        # Find quartiles (25th and 75th percentiles)
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1  # Interquartile range
        
        # Standard statistical rule: outliers are 1.5 * IQR beyond quartiles
        outlier_threshold = 1.5 * iqr
        
        # Find predictions with errors beyond the threshold
        outliers = np.abs(residuals) > outlier_threshold
        
        return {
            'count': np.sum(outliers),                    # How many outliers?
            'percentage': np.mean(outliers) * 100,        # What percentage are outliers?
            'threshold': outlier_threshold                # What threshold was used?
        }
    
    
    def _calculate_prediction_bounds(self, y_pred: np.ndarray, residuals: np.ndarray) -> Dict:
        """
        CALCULATE CONFIDENCE INTERVALS FOR PREDICTIONS
        
        This creates "error bars" around each prediction that tell us:
        "We're X% confident the true value falls within this range"
        
        Uses the standard deviation of residuals to estimate uncertainty:
        - 68% confidence: ± 1 standard deviation
        - 95% confidence: ± 1.96 standard deviations
        
        OUTPUT: Dictionary with upper and lower bounds for each confidence level
        """
        residual_std = np.std(residuals)  # Standard deviation of errors
        
        return {
            # 95% confidence intervals (most commonly used)
            'lower_95': y_pred - 1.96 * residual_std,    # Lower bound (95% confidence)
            'upper_95': y_pred + 1.96 * residual_std,    # Upper bound (95% confidence)
            
            # 68% confidence intervals (1 standard deviation)
            'lower_68': y_pred - residual_std,           # Lower bound (68% confidence)
            'upper_68': y_pred + residual_std            # Upper bound (68% confidence)
        }
    
    
    # STEP 5: MODEL COMPARISON METHODS
    # =============================================================================
    # These methods help compare multiple models against each other
    # Like creating a leaderboard or ranking system
    # =============================================================================
    
    def compare_models(self, 
                      models_results: Dict[str, Dict], 
                      primary_metric: str = 'r2') -> pd.DataFrame:
        """
        COMPARE MULTIPLE MODELS AND CREATE RANKING TABLE
        
        This creates a "leaderboard" showing how different models perform
        relative to each other. Like comparing test scores across students.
        
        INPUT:
        - models_results: Dictionary containing results for each model
        - primary_metric: Which metric to use for ranking ('r2', 'rmse', 'mae')
        
        OUTPUT: Pandas DataFrame with models ranked by performance
        """
        
        comparison_data = []  # List to store comparison information
        
        # EXTRACT KEY METRICS FOR EACH MODEL
        # Loop through each model and extract the most important metrics
        for model_name, results in models_results.items():
            row = {
                'Model': model_name,                                                    # Model name
                'R²': results.get('r2', np.nan),                                      # R-squared score
                'RMSE': results.get('rmse', np.nan),                                  # Root mean squared error
                'MAE': results.get('mae', np.nan),                                    # Mean absolute error
                'MAPE': results.get('mape', np.nan),                                  # Mean absolute percentage error
                'Within_10%': results.get('within_10_percent', np.nan),              # % predictions within 10%
                'Within_20%': results.get('within_20_percent', np.nan),              # % predictions within 20%
                'Outliers_%': results.get('outlier_predictions', {}).get('percentage', np.nan)  # % outlier predictions
            }
            comparison_data.append(row)
        
        # CREATE DATAFRAME FROM COMPARISON DATA
        comparison_df = pd.DataFrame(comparison_data)
        
        # SORT MODELS BY CHOSEN METRIC
        # Different metrics have different "good" directions:
        # - R² (higher is better)
        # - RMSE, MAE (lower is better)
        if primary_metric == 'r2':
            comparison_df = comparison_df.sort_values('R²', ascending=False)      # Higher R² is better
        elif primary_metric == 'rmse':
            comparison_df = comparison_df.sort_values('RMSE', ascending=True)     # Lower RMSE is better
        elif primary_metric == 'mae':
            comparison_df = comparison_df.sort_values('MAE', ascending=True)      # Lower MAE is better
        
        # ADD RANKING COLUMN
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)  # 1st, 2nd, 3rd, etc.
        
        # REORDER COLUMNS FOR BETTER PRESENTATION
        columns = ['Rank', 'Model', 'R²', 'RMSE', 'MAE', 'MAPE', 'Within_10%', 'Within_20%', 'Outliers_%']
        comparison_df = comparison_df[columns]
        
        return comparison_df
    
    
    # STEP 6: REPORT GENERATION METHODS
    # =============================================================================
    # These methods create human-readable reports from the numerical results
    # Like converting grades into report cards with explanations
    # =============================================================================
    
    def generate_model_report(self, model_name: str) -> str:
        """
        GENERATE DETAILED TEXT REPORT FOR A SINGLE MODEL
        
        This creates a comprehensive "report card" for a model that includes:
        - All key performance metrics
        - Interpretation of what the numbers mean
        - Identification of strengths and weaknesses
        - Recommendations for improvement
        
        OUTPUT: Formatted string report that's easy to read and understand
        """
        
        # CHECK IF WE HAVE RESULTS FOR THIS MODEL
        if model_name not in self.evaluation_results:
            return f"No results found for model: {model_name}"
        
        results = self.evaluation_results[model_name]  # Get stored results
        
        # CREATE FORMATTED REPORT HEADER
        report = f"""
{'=' * 50}
MODEL EVALUATION REPORT: {model_name}
{'=' * 50}

PERFORMANCE METRICS:
  R² Score:           {results.get('r2', 'N/A'):.4f}
  Root Mean Sq Error: {results.get('rmse', 'N/A'):.4f}
  Mean Absolute Error: {results.get('mae', 'N/A'):.4f}
  Mean Abs % Error:   {results.get('mape', 'N/A'):.2f}%

PREDICTION ACCURACY:
  Within 10%:         {results.get('within_10_percent', 'N/A'):.1f}% of predictions
  Within 20%:         {results.get('within_20_percent', 'N/A'):.1f}% of predictions
  Within 30%:         {results.get('within_30_percent', 'N/A'):.1f}% of predictions

RESIDUAL ANALYSIS:
  Residual Mean:      {results.get('residual_mean', 'N/A'):.4f}
  Residual Std Dev:   {results.get('residual_std', 'N/A'):.4f}
  Skewness:          {results.get('residual_skewness', 'N/A'):.4f}
  Kurtosis:          {results.get('residual_kurtosis', 'N/A'):.4f}

OUTLIER ANALYSIS:
  Outlier Predictions: {results.get('outlier_predictions', {}).get('count', 'N/A')}
  Outlier Percentage:  {results.get('outlier_predictions', {}).get('percentage', 'N/A'):.1f}%

MODEL INTERPRETATION:
"""
        
        # ADD PERFORMANCE INTERPRETATION BASED ON R² SCORE
        # This translates numerical scores into plain English assessments
        r2 = results.get('r2', 0)
        if r2 >= 0.9:
            report += "  • Excellent model performance\n"
        elif r2 >= 0.8:
            report += "  • Very good model performance\n"
        elif r2 >= 0.7:
            report += "  • Good model performance\n"
        elif r2 >= 0.5:
            report += "  • Moderate model performance\n"
        else:
            report += "  • Poor model performance - consider feature engineering\n"
        
        # ADD ACCURACY INTERPRETATION BASED ON MAPE
        # Lower MAPE = more accurate predictions
        mape = results.get('mape', float('inf'))
        if mape <= 10:
            report += "  • Very accurate predictions (MAPE ≤ 10%)\n"
        elif mape <= 20:
            report += "  • Reasonably accurate predictions (MAPE ≤ 20%)\n"
        elif mape <= 50:
            report += "  • Moderate prediction accuracy (MAPE ≤ 50%)\n"
        else:
            report += "  • Poor prediction accuracy (MAPE > 50%)\n"
        
        # ADD RESIDUAL ANALYSIS INTERPRETATION
        # Check if errors show systematic bias
        skewness = abs(results.get('residual_skewness', 0))
        if skewness > 1:
            report += "  • Residuals show significant skewness - model may have bias\n"
        
        report += "\n" + "=" * 50
        return report
    
    
    # STEP 7: ADVANCED VALIDATION METHODS
    # =============================================================================
    # These methods test how reliable and robust the model performance is
    # Like testing a student multiple times to ensure consistent performance
    # =============================================================================
    
    def cross_validate_model(self, 
                           model,                    # The model to test
                           X: np.ndarray,           # Feature data
                           y: np.ndarray,           # Target data
                           cv: int = 5,             # Number of test rounds
                           scoring: str = 'r2') -> Dict:  # Metric to use
        """
        PERFORM CROSS-VALIDATION ON A MODEL
        
        Cross-validation tests model reliability by:
        1. Splitting data into K parts (folds)
        2. Training on K-1 parts, testing on the remaining part
        3. Repeating K times with different test parts
        4. Averaging results across all rounds
        
        This tells us if the model performs consistently or got lucky on one dataset.
        
        OUTPUT: Statistics about performance consistency across multiple tests
        """
        
        try:
            # Run cross-validation - this trains and tests the model multiple times
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            # CALCULATE RELIABILITY STATISTICS
            cv_results = {
                'scores': scores,                                    # Individual test scores
                'mean_score': np.mean(scores),                      # Average performance
                'std_score': np.std(scores),                        # Performance consistency
                'min_score': np.min(scores),                        # Worst performance
                'max_score': np.max(scores),                        # Best performance
                'confidence_interval_95': (                        # 95% confidence range
                    np.mean(scores) - 1.96 * np.std(scores),      # Lower bound
                    np.mean(scores) + 1.96 * np.std(scores)       # Upper bound
                )
            }
            
            return cv_results
            
        except Exception as e:
            # If cross-validation fails, return error information
            return {'error': f'Cross-validation failed: {str(e)}'}
    
    
    def calculate_learning_curve_metrics(self,
                                       model,                           # Model to analyze
                                       X: np.ndarray,                   # Feature data
                                       y: np.ndarray,                   # Target data
                                       train_sizes: np.ndarray = None) -> Dict:  # Training sizes to test
        """
        ANALYZE HOW MODEL PERFORMANCE CHANGES WITH TRAINING DATA SIZE
        
        This answers questions like:
        - "Would more training data help?"
        - "Is the model overfitting or underfitting?"
        - "What's the optimal amount of training data?"
        
        Method:
        1. Train model with different amounts of data (10%, 20%, 50%, 100%, etc.)
        2. Measure performance at each training size
        3. Plot how performance changes with data size
        
        OUTPUT: Information about optimal training size and learning patterns
        """
        
        # Import learning curve function
        from sklearn.model_selection import learning_curve
        
        # Set default training sizes if not provided
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)  # 10%, 20%, ..., 100% of data
        
        try:
            # Calculate learning curves
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=5, scoring='r2'
            )
            
            # ANALYZE LEARNING CURVE RESULTS
            results = {
                'train_sizes': train_sizes_abs,                      # Actual training sizes used
                'train_scores_mean': np.mean(train_scores, axis=1),  # Average training performance
                'train_scores_std': np.std(train_scores, axis=1),    # Training performance variation
                'val_scores_mean': np.mean(val_scores, axis=1),      # Average validation performance
                'val_scores_std': np.std(val_scores, axis=1),        # Validation performance variation
                'optimal_size_index': np.argmax(np.mean(val_scores, axis=1)),  # Best training size
                'gap_at_max': np.mean(train_scores, axis=1)[-1] - np.mean(val_scores, axis=1)[-1]  # Overfitting gap
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Learning curve calculation failed: {str(e)}'}
    
    
    # STEP 8: MODEL DIAGNOSIS METHODS
    # =============================================================================
    # These methods identify specific problems and suggest solutions
    # Like a doctor diagnosing what's wrong and recommending treatment
    # =============================================================================
    
    def diagnose_model_issues(self, model_name: str) -> List[str]:
        """
        DIAGNOSE POTENTIAL ISSUES WITH MODEL PERFORMANCE
        
        This method acts like a "model doctor" that:
        1. Examines all the performance metrics
        2. Identifies specific problems or weaknesses
        3. Provides actionable recommendations for improvement
        
        OUTPUT: List of diagnostic messages and specific recommendations
        """
        
        # CHECK IF WE HAVE RESULTS FOR THIS MODEL
        if model_name not in self.evaluation_results:
            return [f"No evaluation results found for {model_name}"]
        
        results = self.evaluation_results[model_name]
        issues = []  # List to store identified issues and recommendations
        
        # DIAGNOSIS 1: LOW R² SCORE (Poor Model Fit)
        r2 = results.get('r2', 0)
        if r2 < 0.3:
            issues.append("⚠️ Very low R² score suggests poor model fit. Consider:")
            issues.append("   • Adding more relevant features")
            issues.append("   • Feature engineering or transformation")
            issues.append("   • Trying non-linear models")
        elif r2 < 0.5:
            issues.append("⚠️ Low R² score. Model explains less than 50% of variance.")
        
        # DIAGNOSIS 2: HIGH PREDICTION ERRORS
        mape = results.get('mape', 0)
        if mape > 50:
            issues.append("⚠️ Very high prediction errors (MAPE > 50%). Consider:")
            issues.append("   • Data quality checks")
            issues.append("   • Outlier removal")
            issues.append("   • Feature scaling")
        
        # DIAGNOSIS 3: BIASED PREDICTIONS (Skewed Residuals)
        residual_skewness = abs(results.get('residual_skewness', 0))
        if residual_skewness > 1:
            issues.append("⚠️ Skewed residuals suggest model bias. Consider:")
            issues.append("   • Target variable transformation (log, sqrt)")
            issues.append("   • Non-linear models")
            issues.append("   • Adding interaction terms")
        
        # DIAGNOSIS 4: TOO MANY OUTLIER PREDICTIONS
        outlier_pct = results.get('outlier_predictions', {}).get('percentage', 0)
        if outlier_pct > 15:
            issues.append(f"⚠️ High percentage of outlier predictions ({outlier_pct:.1f}%):")
            issues.append("   • Check for data quality issues")
            issues.append("   • Consider robust regression techniques")
            issues.append("   • Investigate extreme values in training data")
        
        # DIAGNOSIS 5: POOR PRACTICAL ACCURACY
        within_20_pct = results.get('within_20_percent', 0)
        if within_20_pct < 70:
            issues.append(f"⚠️ Only {within_20_pct:.1f}% predictions within 20% accuracy:")
            issues.append("   • Model may be underfitting")
            issues.append("   • Consider ensemble methods")
            issues.append("   • Increase model complexity")
        
        # IF NO ISSUES FOUND, PROVIDE POSITIVE FEEDBACK
        if not issues:
            issues.append("✅ No major issues detected with model performance")
        
        return issues


# STEP 9: ADVANCED MODEL ANALYZER CLASS
# =============================================================================
# This is a specialized tool for more advanced analysis techniques
# Think of it as the "expert consultant" that provides deeper insights
# =============================================================================

class AdvancedModelAnalyzer:
    """
    ADVANCED ANALYSIS TOOLS FOR MODEL INTERPRETATION
    
    This class provides sophisticated analysis methods that go beyond basic
    performance metrics. It helps answer questions like:
    - Which features work together?
    - How stable are the predictions?
    - What are the model's blind spots?
    """
    
    def __init__(self):
        """Initialize storage for analysis results"""
        self.analysis_results = {}
    
    
    def analyze_feature_interactions(self, 
                                   model,                        # Trained model
                                   X: pd.DataFrame,             # Feature data
                                   feature_names: List[str],    # Names of features
                                   max_interactions: int = 10) -> Dict:  # Maximum interactions to analyze
        """
        ANALYZE HOW FEATURES WORK TOGETHER IN THE MODEL
        
        Some features might be more powerful when combined with others.
        This method identifies which features have the strongest influence
        and how they might interact.
        
        Note: This is a simplified version. Full implementation would require
        more sophisticated interaction detection methods.
        
        OUTPUT: Information about top features and their importance
        """
        
        interactions = {}
        
        # ANALYZE FEATURE IMPORTANCE (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get indices of most important features
            top_features_idx = np.argsort(importances)[-max_interactions:]
            
            # Convert indices to feature names
            top_features = [feature_names[i] for i in top_features_idx]
            
            # Store results
            interactions['top_features'] = top_features
            interactions['importance_scores'] = {
                feature_names[i]: importances[i] for i in top_features_idx
            }
        
        return interactions
    
    
    def calculate_prediction_stability(self,
                                     model,                    # Trained model
                                     X: np.ndarray,           # Data for prediction
                                     n_bootstrap: int = 100) -> Dict:  # Number of stability tests
        """
        CALCULATE HOW STABLE/CONSISTENT THE MODEL'S PREDICTIONS ARE
        
        This method tests prediction stability by:
        1. Creating many different random samples of the data
        2. Making predictions on each sample
        3. Measuring how much predictions vary across samples
        
        Stable predictions = reliable model
        Unstable predictions = model might be overfitting or unreliable
        
        OUTPUT: Stability metrics including consistency scores
        """
        
        predictions = []  # Store predictions from each bootstrap sample
        
        # GENERATE BOOTSTRAP PREDICTIONS
        # Bootstrap = sampling with replacement (like drawing names from a hat, 
        # putting each name back before drawing the next one)
        for _ in range(n_bootstrap):
            # Create random sample of same size as original data
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]  # Bootstrap sample
            
            try:
                # Make predictions on this sample
                pred = model.predict(X_boot)
                predictions.append(pred)
            except:
                # Skip if prediction fails
                continue
        
        # CALCULATE STABILITY METRICS
        if predictions:
            predictions = np.array(predictions)
            
            stability_results = {
                'mean_predictions': np.mean(predictions, axis=0),     # Average prediction for each data point
                'std_predictions': np.std(predictions, axis=0),       # How much predictions vary
                'cv_predictions': np.std(predictions, axis=0) / np.mean(predictions, axis=0),  # Coefficient of variation
                'stability_score': 1 - np.mean(np.std(predictions, axis=0) / np.mean(predictions, axis=0))  # Overall stability (higher = more stable)
            }
            
            return stability_results
        
        return {'error': 'Could not calculate stability metrics'}


# STEP 10: UTILITY FUNCTIONS
# =============================================================================
# These are helper functions that create useful summaries and reports
# Think of them as "report generators" that make the results easy to understand
# =============================================================================

def create_evaluation_summary(evaluator: ModelEvaluator, 
                            model_names: List[str]) -> str:
    """
    CREATE A COMPREHENSIVE EVALUATION SUMMARY FOR MULTIPLE MODELS
    
    This function creates an executive summary that includes:
    - Model rankings and comparisons
    - Key performance highlights for each model
    - Recommendations for which model to use
    
    Perfect for presenting results to stakeholders or decision-makers.
    
    INPUT:
    - evaluator: ModelEvaluator instance with stored results
    - model_names: List of model names to include in summary
    
    OUTPUT: Formatted summary string with emojis and clear formatting
    """
    
    # CREATE SUMMARY HEADER
    summary = """
🎯 MODEL EVALUATION SUMMARY
""" + "="*50 + "\n\n"
    
    # SECTION 1: OVERALL COMPARISON TABLE
    # Show side-by-side comparison if we have multiple models
    if len(model_names) > 1:
        comparison_df = evaluator.compare_models(evaluator.evaluation_results)
        summary += "📊 MODEL RANKING:\n"
        summary += comparison_df.to_string(index=False, float_format='%.3f')
        summary += "\n\n"
    
    # SECTION 2: INDIVIDUAL MODEL SUMMARIES
    # Provide quick overview of each model's performance
    for model_name in model_names:
        if model_name in evaluator.evaluation_results:
            results = evaluator.evaluation_results[model_name]
            
            # Model name and key metrics
            summary += f"📈 {model_name.upper()}:\n"
            summary += f"   R² Score: {results.get('r2', 'N/A'):.3f} | "
            summary += f"RMSE: {results.get('rmse', 'N/A'):.3f} | "
            summary += f"Accuracy (±20%): {results.get('within_20_percent', 'N/A'):.1f}%\n"
            
            # QUICK PERFORMANCE INTERPRETATION
            # Translate R² score into plain English assessment
            r2 = results.get('r2', 0)
            if r2 >= 0.8:
                summary += "   ✅ Strong predictive performance\n"
            elif r2 >= 0.6:
                summary += "   ⚠️ Moderate predictive performance\n"
            else:
                summary += "   ❌ Weak predictive performance\n"
            
            summary += "\n"
    
    # SECTION 3: RECOMMENDATION
    # Identify and recommend the best performing model
    if len(model_names) > 1:
        # Find model with highest R² score
        best_model = max(model_names, 
                        key=lambda x: evaluator.evaluation_results.get(x, {}).get('r2', -1))
        summary += f"🏆 RECOMMENDED MODEL: {best_model}\n"
        summary += f"   Best overall performance based on R² score\n\n"
    
    summary += "="*50
    
    return summary


# STEP 11: EXAMPLE USAGE AND TESTING
# =============================================================================
# This section shows how to use the ModelEvaluator in practice
# Think of it as a "quick start guide" or "demo"
# =============================================================================

if __name__ == "__main__":
    """
    EXAMPLE USAGE OF THE MODEL EVALUATOR
    
    This section demonstrates how to use the ModelEvaluator to:
    1. Create sample data
    2. Evaluate multiple models
    3. Compare their performance
    4. Generate reports
    
    This code only runs when the file is executed directly (not imported)
    """
    
    # STEP 1: CREATE MODEL EVALUATOR INSTANCE
    evaluator = ModelEvaluator()
    
    # STEP 2: CREATE SAMPLE DATA FOR TESTING
    # This simulates having real model predictions to evaluate
    
    np.random.seed(42)  # Set random seed for reproducible results
    
    # Create "true" values that we're trying to predict
    y_true = np.random.normal(100, 20, 100)  # 100 values, mean=100, std=20
    
    # Create predictions from two different models:
    # Model A: Good model (small errors)
    y_pred1 = y_true + np.random.normal(0, 5, 100)   # True values + small random errors
    
    # Model B: Poor model (large errors)  
    y_pred2 = y_true + np.random.normal(0, 15, 100)  # True values + large random errors
    
    # STEP 3: EVALUATE BOTH MODELS
    print("Evaluating models...")
    results1 = evaluator.evaluate_model('Model_A', y_true, y_pred1)
    results2 = evaluator.evaluate_model('Model_B', y_true, y_pred2)
    
    # STEP 4: COMPARE MODELS
    print("\nModel Comparison:")
    comparison = evaluator.compare_models(evaluator.evaluation_results)
    print(comparison)
    
    # STEP 5: GENERATE DETAILED REPORT
    print("\nDetailed Report for Best Model:")
    print(evaluator.generate_model_report('Model_A'))