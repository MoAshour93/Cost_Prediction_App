# Project Cost Prediction ML App

A comprehensive machine learning application for construction project cost estimation and predictive analytics. This application uses multiple ML algorithms to predict 'Estimate at Completion' (EAC) values based on various project parameters and historical data.

**Developed by**: Mohamed Ashour - Construction Digital Transformation Manager  
**YouTube Channel**: [APC Mastery Path](https://www.youtube.com/@APCMasteryPath)  
**Website**: [www.apcmasterypath.co.uk](https://www.apcmasterypath.co.uk)  
**LinkedIn**: [Mohamed Ashour](https://www.linkedin.com/in/mohamed-ashour-0727/)

## 📑 Table of Contents

- [🚀 Features](#-features)
- [🏗️ Application Workflow](#️-application-workflow)
- [📋 Requirements](#-requirements)
- [🛠️ Installation](#️-installation)
- [🏃 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [🔧 Configuration](#-configuration)
- [🧪 Advanced Features](#-advanced-features)
- [📊 Model Performance](#-model-performance)
- [🔍 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📞 Support](#-support)
- [🔄 Version History](#-version-history)

## 🚀 Features

### Core Functionality
- **Multi-Model Support**: Linear Regression, Random Forest, XGBoost, and additional ensemble methods
- **Intelligent Data Processing**: Automatic data type detection and preprocessing
- **Batch Processing**: Memory-efficient processing for large datasets
- **Real-time Predictions**: Single project predictions with confidence intervals
- **Comprehensive Validation**: Data quality checks and validation pipeline

### Data Support
- **Multiple File Formats**: CSV, Excel (.xlsx, .xls, .xlsb)
- **Smart Data Handling**: Automatic outlier detection and removal
- **Feature Engineering**: Categorical encoding and numerical scaling
- **Large File Support**: Memory-efficient chunk processing

### Analysis & Visualization
- **Performance Metrics**: R², MSE, MAE, MAPE analysis
- **Feature Importance**: Model-specific importance ranking
- **Data Visualization**: Correlation matrices, scatter plots, residual analysis
- **Model Comparison**: Side-by-side model performance evaluation

### User Interface
- **Intuitive GUI**: Tkinter-based graphical interface
- **Progress Tracking**: Real-time processing updates
- **Export Capabilities**: Save predictions and analysis results
- **Error Handling**: Comprehensive error reporting and recovery

## 🏗️ Application Workflow

### Overview
The Project Estimation ML App follows a systematic workflow designed to maximize prediction accuracy while maintaining ease of use for construction professionals.

### Phase 1: Data Import and Validation
```
📁 Load Data → 🔍 Validate Format → ✅ Quality Check → 📊 Data Preview
```

**Step 1.1: File Loading**
- Support for multiple formats (CSV, Excel .xlsx/.xls/.xlsb)
- Automatic file size validation (max 500MB)
- Encoding detection and handling
- Memory-efficient loading for large datasets

**Step 1.2: Data Structure Validation**
- Column header detection and mapping
- Target variable identification (`Estimate_at_Completion`)
- Feature type classification (numeric/categorical)
- Missing value assessment and reporting

**Step 1.3: Quality Assessment**
- Data completeness analysis
- Outlier detection using IQR method
- Statistical distribution analysis
- Data consistency checks

### Phase 2: Data Preprocessing and Feature Engineering
```
🔧 Clean Data → 🏷️ Encode Features → 📏 Scale Values → 🎯 Prepare Target
```

**Step 2.1: Data Cleaning**
- Outlier removal (configurable thresholds)
- Missing value imputation strategies
- Data type corrections and conversions
- Duplicate record identification

**Step 2.2: Feature Engineering**
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Numerical Scaling**: StandardScaler for feature normalization  
- **Feature Selection**: Correlation analysis and importance ranking
- **Data Splitting**: Train/test split (default 80/20, configurable)

**Step 2.3: Target Variable Processing**
- Ensure numeric target values
- Handle extreme values appropriately
- Log transformation if needed for skewed distributions

### Phase 3: Model Training and Selection
```
🤖 Train Models → 📈 Evaluate Performance → 🏆 Select Best → 🔍 Analyze Features
```

**Step 3.1: Multi-Model Training**
The application trains multiple algorithms simultaneously:
- **Linear Regression**: Fast baseline with feature interpretability
- **Random Forest**: Robust ensemble handling non-linear relationships
- **XGBoost**: High-performance gradient boosting
- **Additional Models**: Ridge, Lasso, ElasticNet (configurable)

**Step 3.2: Performance Evaluation**
- **Cross-validation**: 5-fold validation for robust metrics
- **Multiple Metrics**: R², MSE, RMSE, MAE, MAPE
- **Residual Analysis**: Pattern detection in prediction errors
- **Overfitting Detection**: Training vs validation performance comparison

**Step 3.3: Model Comparison and Selection**
- Side-by-side performance comparison
- Statistical significance testing
- Feature importance analysis
- Model complexity vs accuracy trade-off

### Phase 4: Prediction and Analysis
```
🎯 Make Predictions → 📊 Visualize Results → 📋 Generate Reports → 💾 Export Data
```

**Step 4.1: Prediction Generation**
- **Single Project Predictions**: Input new project parameters
- **Batch Processing**: Handle multiple projects simultaneously
- **Confidence Intervals**: Prediction uncertainty quantification
- **Sensitivity Analysis**: Impact of feature changes

**Step 4.2: Results Visualization**
- **Performance Charts**: Model accuracy comparisons
- **Feature Importance Plots**: Top predictive factors
- **Correlation Matrices**: Feature relationship analysis
- **Residual Plots**: Error pattern identification
- **Prediction vs Actual Scatter Plots**: Model accuracy visualization

**Step 4.3: Business Intelligence**
- **Cost Driver Analysis**: Identify primary cost factors
- **Risk Assessment**: Flag high-uncertainty predictions
- **Scenario Modeling**: What-if analysis capabilities
- **Trend Analysis**: Historical pattern identification

### Phase 5: Export and Integration
```
📤 Export Predictions → 📑 Generate Reports → 🔄 Model Persistence → 🔗 API Integration
```

**Step 5.1: Data Export Options**
- CSV format for spreadsheet compatibility
- Excel workbooks with multiple sheets
- Formatted reports with visualizations


### Workflow Benefits for Construction Professionals

**Time Efficiency**
- Automated data processing reduces manual effort
- Batch prediction capabilities for portfolio analysis
- Quick model retraining with new data

**Accuracy Improvements**
- Multiple model approach reduces bias
- Continuous validation ensures reliability
- Domain-specific feature engineering

**Decision Support**
- Clear visualization of cost drivers
- Confidence intervals for risk assessment
- Scenario analysis for project planning

**Integration Ready**
- Standard export formats for existing workflows
- API-ready predictions for enterprise systems
- Scalable architecture for growing datasets

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Core Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
xgboost>=1.6.0
scipy>=1.8.0
```

### Excel Support Dependencies
```
openpyxl>=3.0.0      # For .xlsx files
pyxlsb>=1.0.0        # For .xlsb files  
xlrd>=2.0.0          # For legacy .xls files
```

### Optional Dependencies
```
joblib>=1.1.0        # Enhanced performance
```

## 🛠️ Installation

### Quick Start
1. **Clone or download** the project files
2. **Install dependencies** using one of these methods:

#### Method 1: Automatic Installation (Recommended)
```bash
python install.py
```

#### Method 2: Manual Installation
```bash
pip install -r Requirements.txt
```

#### Method 3: Individual Package Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy openpyxl pyxlsb xlrd joblib
```

### Verify Installation
```bash
python run_app.py
```

## 🏃 Usage

### Starting the Application
```bash
python run_app.py
```

### Basic Workflow
1. **Load Data**: Import your project data (CSV or Excel)
2. **Data Processing**: Review and process data with automatic validation
3. **Model Training**: Train multiple ML models on your dataset
4. **Make Predictions**: Generate predictions for new projects
5. **Analysis**: Review model performance and feature importance
6. **Export Results**: Save predictions and analysis

### Supported Data Format
Your dataset should include:
- **Target Variable**: `Estimate_at_Completion` (numeric)
- **Features**: Any combination of numeric and categorical variables
- **Minimum**: 10 samples for training
- **Recommended**: 100+ samples for reliable predictions

### Example Data Structure
```csv
Project_Size,Complexity,Location,Material_Cost,Labor_Hours,Estimate_at_Completion
1000,High,Urban,75000,2500,180000
800,Medium,Rural,50000,1800,120000
1200,Low,Suburban,60000,2000,140000
```

## 📁 Project Structure

```
Cost Prediction App v1.2.2/
├── project_estimation_app.py    # Main GUI application
├── run_app.py                   # Application launcher with validation
├── config.py                    # Configuration settings
├── batch_processor.py           # Batch processing utilities
├── data_validator.py            # Data validation functions
├── model_evaluator.py           # Model evaluation and metrics
├── debug_prediction.py          # Debugging and testing utilities
├── install.py                   # Automated installation script
├── Requirements.txt             # Python dependencies
└── logs/                        # Application logs
```

## 🔧 Configuration

The application can be customized through `config.py`:

### Model Parameters
- Default Random Forest settings (n_estimators, max_depth, etc.)
- XGBoost configuration (learning_rate, max_depth, etc.)
- Linear Regression options

### Data Processing
- Training/testing split ratios (default: 80/20)
- Outlier detection thresholds
- Missing value handling

### Performance Settings
- Maximum file size limits
- Chunk size for large datasets
- Display row limits

### Visualization
- Plot styles and color schemes
- Figure sizes and DPI settings
- Maximum features in plots

## 🧪 Advanced Features

### Batch Processing
Process large datasets efficiently:
```python
from batch_processor import BatchProcessor
processor = BatchProcessor(chunk_size=5000)
```

### Model Evaluation
Comprehensive model analysis:
```python
from model_evaluator import ModelEvaluator
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model_name, y_true, y_pred)
```

### Data Validation
Validate data quality:
```python
from data_validator import DataValidator
validator = DataValidator()
validation_results = validator.validate_dataset(df)
```

### Debug Mode
Test predictions without GUI:
```python
python debug_prediction.py
```

## 📊 Model Performance

The application supports multiple regression models:

- **Linear Regression**: Fast baseline model
- **Random Forest**: Robust ensemble method
- **XGBoost**: High-performance gradient boosting
- **Additional Models**: Ridge, Lasso, ElasticNet, SVM, KNN

Performance metrics include:
- R² Score (coefficient of determination)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

## 🔍 Troubleshooting

### Common Issues

#### Dependencies Not Found
```bash
pip install -r Requirements.txt
# or run the installation script
python install.py
```

#### GUI Not Starting
- Ensure tkinter is installed (usually comes with Python)
- Check Python version (3.7+ required)
- Review logs in `logs/` directory

#### Memory Issues with Large Files
- Use batch processing mode
- Reduce chunk size in config
- Close other applications to free memory

#### Prediction Errors
- Verify target variable is numeric
- Check for missing values in features
- Ensure sufficient training data (10+ samples)

### Log Files
Check `logs/` directory for detailed error information:
- `app_launch_*.log`: Application startup logs
- `claude_code_*.log`: Development session logs

## 🤝 Contributing

This project is designed for construction industry cost estimation and welcomes collaborative contributions. However, please note the licensing terms below before contributing.

**Areas for contribution:**
- Model accuracy improvements
- Additional data validation features
- Performance optimizations  
- User interface enhancements
- Documentation improvements
- Bug fixes and testing
- **Model Persistence**
  - Save trained models for reuse
  - Version control for model updates
  - Performance tracking over time
  - Model lifecycle management


**How to contribute:**
1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description
5. Engage in code review process

All contributions must align with the project's commercial licensing terms.

## 📄 License

**Commercial Use License**

This software is proprietary and developed by Mohamed Ashour. While the source code is available for viewing and collaborative development, it is **not free for commercial use**.

### Usage Terms:
- ✅ **Personal/Educational Use**: Free for learning and non-commercial research
- ✅ **Evaluation**: Free 14-day trial for commercial evaluation
- ❌ **Commercial Use**: Requires paid license for business/commercial applications
- ❌ **Redistribution**: Cannot redistribute or sell without permission

### Why Not Free?
This application represents:
- Substantial time in development and testing using various ML algorithms
- Extensive knowledge and experience in the construction industry
- Advanced data processing techniques
- Customized ML models tailored to construction needs
- Comprehensive validation and testing
- Extensive documentation and support
- Ongoing maintenance and improvements

### Commercial Licensing:
For commercial use, please contact:
- **Email**: Available via [LinkedIn](https://www.linkedin.com/in/mohamed-ashour-0727/)
- **Website**: [www.apcmasterypath.co.uk](https://www.apcmasterypath.co.uk)
- **YouTube**: [APC Mastery Path](https://www.youtube.com/@APCMasteryPath)

### Academic/Research Use:
Academic institutions and researchers can use this software for non-commercial purposes. Please cite appropriately in any publications.

## 📞 Support

For technical issues:
1. Check the `logs/` directory for error details
2. Review this README for troubleshooting steps
3. Verify all dependencies are installed correctly
4. Ensure your data format matches the expected structure

## 🔄 Version History

- **v1.2.4**: Enhanced batch processing and model evaluation
- **v1.2.3**: Current version with improved GUI and validation
- **v1.2.2**: Added enhancements for the processing of data in addition to the graphs in the visualisations pane.
- **v1.2.1**: Older release relying on a web-app structure making the best use of React JS.
- **v1.1.0**: Enhanced Machine Learning Functionality & Outlier Detection
- **v1.0.0**: Initial App with a minimalistic front-end and basic ML Functionality

---

**Note**: This application is designed for construction project cost estimation and should be used with domain expertise for optimal results.
