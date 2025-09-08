#!/usr/bin/env python3
# ==============================================================================
# AUTOMATED INSTALLATION AND SETUP SCRIPT
# ==============================================================================
#
# PURPOSE: Complete automated setup system for the Project Estimation ML App
#
# WHAT THIS SCRIPT DOES:
# - Automatically installs all required software dependencies
# - Sets up the proper directory structure for the application
# - Creates sample data for immediate testing and training
# - Validates that everything is working correctly
# - Generates detailed installation reports for troubleshooting
#
# BUSINESS BENEFITS:
# - Eliminates manual setup errors and reduces deployment time
# - Ensures consistent installation across different computers/environments
# - Provides immediate validation that the system is working
# - Creates sample data so users can start immediately
# - Generates audit trail for IT compliance and troubleshooting
#
# TARGET USERS:
# - IT administrators deploying the application
# - End users setting up the system on their computers
# - Developers setting up development environments
# - System administrators managing multiple installations
#
# REAL-WORLD ANALOGY:
# Like an automated factory setup that installs all the machinery,
# tests that everything works, and produces the first sample products
# to prove the entire production line is ready for business
# ==============================================================================

# Automated installation and setup script for Project Estimation ML App
# Handles dependency installation, environment setup, and initial configuration

# STEP 1: IMPORT SYSTEM LIBRARIES
# These provide access to system functions and external program execution
import subprocess      # Run external commands (like pip install)
import sys            # Python system information and control
import os             # Operating system interface
import platform       # Platform identification (Windows, Mac, Linux)
import json           # JSON file creation and parsing
from pathlib import Path        # Modern file and directory handling
import importlib.util          # Dynamic module importing for testing


# ==============================================================================
# MAIN INSTALLATION MANAGER CLASS
# ==============================================================================
class InstallationManager:
    """
    COMPREHENSIVE INSTALLATION SYSTEM
    
    WHAT IT MANAGES:
    - System compatibility checking (Python version, pip availability)
    - Automated dependency installation with verification
    - Directory structure creation for organized file storage
    - Sample data generation for immediate testing
    - Configuration file creation with sensible defaults
    - Comprehensive testing to ensure everything works
    - Detailed reporting for troubleshooting and audit trails
    
    BUSINESS VALUE:
    - Eliminates the "it works on my machine" problem
    - Reduces setup time from hours to minutes
    - Provides consistent, repeatable installations
    - Catches compatibility issues before they cause problems
    - Creates professional installation experience
    
    HOW IT WORKS:
    1. Checks system prerequisites (Python version, pip)
    2. Installs all required Python packages
    3. Creates necessary directories and files
    4. Runs comprehensive functionality tests
    5. Generates detailed installation report
    """
    
    def __init__(self):
        """
        INITIALIZATION: Set up installation tracking and system information
        
        These variables track the installation process and system details:
        - python_version: Current Python version for compatibility checking
        - platform: Operating system (Windows, macOS, Linux)
        - required_python_version: Minimum Python version needed
        - installation_log: Complete record of all installation steps
        - failed_packages: List of packages that failed to install
        """
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.required_python_version = (3, 7)    # Minimum Python 3.7 required
        self.installation_log = []                # Track all installation steps
        self.failed_packages = []                 # Track any installation failures
        
    # --------------------------------------------------------------------------
    # LOGGING AND PROGRESS TRACKING
    # --------------------------------------------------------------------------
    def log(self, message):
        """
        PROGRESS LOGGING: Track and display installation progress
        
        BUSINESS PURPOSE:
        - Provides real-time feedback to users during installation
        - Creates audit trail for troubleshooting and compliance
        - Helps identify where problems occur if installation fails
        
        This function both displays messages to the user and stores them
        for the final installation report
        """
        print(message)                           # Show user immediate feedback
        self.installation_log.append(message)   # Store for final report
    
    # --------------------------------------------------------------------------
    # SYSTEM COMPATIBILITY CHECKS
    # --------------------------------------------------------------------------
    def check_python_version(self):
        """
        PYTHON VERSION VALIDATION: Ensure compatible Python version
        
        BUSINESS IMPORTANCE:
        The application requires modern Python features and libraries that
        aren't available in older versions. Using incompatible versions
        leads to cryptic errors and frustrated users.
        
        COMPATIBILITY REQUIREMENTS:
        - Python 3.7+: Required for modern pandas, scikit-learn, and type hints
        - Older versions lack security updates and modern features
        - Machine learning libraries require recent Python versions
        
        Returns: True if compatible, False if upgrade needed
        """
        self.log("Checking Python version...")
        
        # Compare current version with minimum requirement
        if self.python_version < self.required_python_version:
            self.log(f"❌ Python {self.required_python_version[0]}.{self.required_python_version[1]}+ required")
            self.log(f"   Current version: {sys.version}")
            return False
        
        # Version is compatible
        self.log(f"✅ Python version {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} is compatible")
        return True
    
    def check_pip(self):
        """
        PIP AVAILABILITY CHECK: Ensure package installer is working
        
        BUSINESS IMPORTANCE:
        pip is Python's package installer - without it, we can't install
        the machine learning libraries needed for the application
        
        WHAT THIS CHECKS:
        - Is pip installed and accessible?
        - Can we run pip commands programmatically?
        - Is pip properly configured with Python?
        
        Returns: True if pip is available, False if not
        """
        self.log("Checking pip availability...")
        
        try:
            # Try to run 'pip --version' command silently
            subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log("✅ pip is available")
            return True
        except subprocess.CalledProcessError:
            self.log("❌ pip is not available")
            return False
    
    def upgrade_pip(self):
        """
        PIP UPGRADE: Ensure latest pip version for best compatibility
        
        BUSINESS REASONING:
        - Newer pip versions have better dependency resolution
        - Fixes compatibility issues with modern packages
        - Improves installation success rates
        - Reduces cryptic error messages during package installation
        
        This is not critical for installation but improves reliability
        """
        self.log("Upgrading pip...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log("✅ pip upgraded successfully")
            return True
        except subprocess.CalledProcessError:
            self.log("⚠️ Failed to upgrade pip, continuing with current version")
            return False
    
    # --------------------------------------------------------------------------
    # PACKAGE INSTALLATION SYSTEM
    # --------------------------------------------------------------------------
    def install_package(self, package_name, package_spec=None):
        """
        INDIVIDUAL PACKAGE INSTALLER: Install and verify single packages
        
        INSTALLATION PROCESS:
        1. Check if package is already installed (avoid unnecessary work)
        2. Install package using pip with specific version requirements
        3. Verify installation was successful
        4. Track any failures for troubleshooting
        
        BUSINESS BENEFITS:
        - Avoids reinstalling existing packages (saves time)
        - Uses specific version requirements for consistency
        - Verifies each installation to catch problems early
        - Provides detailed feedback on progress and failures
        
        Parameters:
        - package_name: Name used for import testing (e.g., 'sklearn')
        - package_spec: Full pip specification (e.g., 'scikit-learn>=1.1.0')
        
        Returns: True if successful, False if failed
        """
        install_name = package_spec or package_name
        
        # STEP 1: Check if already installed
        if self.is_package_installed(package_name):
            self.log(f"✅ {package_name} is already installed")
            return True
        
        # STEP 2: Install the package
        self.log(f"Installing {package_name}...")
        
        try:
            # Run pip install command silently
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", install_name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # STEP 3: Verify installation worked
            if self.is_package_installed(package_name):
                self.log(f"✅ {package_name} installed successfully")
                return True
            else:
                self.log(f"❌ {package_name} installation verification failed")
                self.failed_packages.append(package_name)
                return False
                
        except subprocess.CalledProcessError as e:
            # Handle installation failures gracefully
            self.log(f"❌ Failed to install {package_name}: {e}")
            self.failed_packages.append(package_name)
            return False
    
    def is_package_installed(self, package_name):
        """
        PACKAGE VERIFICATION: Check if a package is properly installed
        
        SPECIAL CASES HANDLED:
        - tkinter: Built into Python but import name differs
        - sklearn: Package name is 'scikit-learn' but import is 'sklearn'
        - Standard packages: Use standard import mechanism
        
        This prevents false negatives where packages are installed
        but have different import names than their pip names
        
        Returns: True if package can be imported, False if not
        """
        try:
            # Handle special cases with different import names
            if package_name == "tkinter":
                import tkinter
            elif package_name == "sklearn":
                import sklearn
            else:
                # Standard import test
                importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def install_requirements(self):
        """
        BULK PACKAGE INSTALLATION: Install all required dependencies
        
        PACKAGE SELECTION STRATEGY:
        - Core data science stack: pandas, numpy, matplotlib, seaborn
        - Machine learning: scikit-learn, xgboost, scipy
        - File handling: openpyxl for Excel files
        - Utilities: joblib for model persistence
        - GUI: tkinter (usually built-in)
        
        VERSION SPECIFICATIONS:
        - Minimum versions ensure compatibility with application features
        - Not maximum versions to allow users to upgrade if desired
        - Versions chosen based on stability and feature requirements
        
        BUSINESS IMPACT:
        - Ensures all features work correctly
        - Prevents version compatibility issues
        - Provides consistent environment across installations
        """
        self.log("\\n" + "="*50)
        self.log("INSTALLING REQUIRED PACKAGES")
        self.log("="*50)
        
        # DEFINE PACKAGE REQUIREMENTS
        # Map import names to pip package specifications
        packages = {
            'pandas': 'pandas>=1.5.0',           # Data manipulation and analysis
            'numpy': 'numpy>=1.21.0',            # Numerical computing foundation
            'matplotlib': 'matplotlib>=3.5.0',   # Basic plotting and visualization
            'seaborn': 'seaborn>=0.11.0',       # Statistical visualization
            'sklearn': 'scikit-learn>=1.1.0',   # Machine learning algorithms
            'xgboost': 'xgboost>=1.6.0',        # Advanced gradient boosting
            'scipy': 'scipy>=1.8.0',            # Scientific computing utilities
            'joblib': 'joblib>=1.1.0',          # Model persistence and parallel processing
            'openpyxl': 'openpyxl>=3.0.0'       # Excel file reading/writing
        }
        
        successful_installs = 0
        
        # INSTALL EACH PACKAGE
        for package_name, package_spec in packages.items():
            if self.install_package(package_name, package_spec):
                successful_installs += 1
        
        # SPECIAL CHECK FOR TKINTER
        # tkinter is usually built into Python but may be missing on some systems
        if self.is_package_installed('tkinter'):
            self.log("✅ tkinter is available")
            successful_installs += 1
        else:
            self.log("⚠️ tkinter not available - GUI may not work")
            self.log("   Install tkinter: sudo apt-get install python3-tk (Ubuntu/Debian)")
            self.log("   or: brew install tkinter (macOS with Homebrew)")
        
        # INSTALLATION SUMMARY
        self.log(f"\\nInstallation Summary: {successful_installs}/{len(packages)+1} packages successful")
        
        if self.failed_packages:
            self.log(f"Failed packages: {', '.join(self.failed_packages)}")
            return False
        
        return successful_installs >= len(packages)
    
    # --------------------------------------------------------------------------
    # DIRECTORY STRUCTURE SETUP
    # --------------------------------------------------------------------------
    def create_directory_structure(self):
        """
        PROJECT DIRECTORY CREATION: Set up organized file structure
        
        DIRECTORY PURPOSE:
        - data/: Store input datasets and sample files
        - models/: Save trained machine learning models
        - outputs/: Store prediction results and reports
        - logs/: Application logs and error tracking
        - exports/: User-exported files and results
        
        BUSINESS BENEFITS:
        - Organizes files for easy management
        - Prevents file clutter and confusion
        - Supports backup and maintenance procedures
        - Follows professional software organization standards
        """
        self.log("\\nCreating directory structure...")
        
        directories = [
            'data',      # Input data files and samples
            'models',    # Trained ML models
            'outputs',   # Prediction results
            'logs',      # Application logs
            'exports'    # User exports
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(exist_ok=True)  # Create if doesn't exist
                self.log(f"✅ Created directory: {directory}/")
            except Exception as e:
                self.log(f"❌ Failed to create directory {directory}: {e}")
    
    def create_sample_data(self):
        """
        SAMPLE DATA GENERATION: Create realistic test data for immediate use
        
        BUSINESS PURPOSE:
        - Users can start using the application immediately
        - Provides realistic data for training and testing
        - Demonstrates expected data format and structure
        - Enables validation that the system works correctly
        
        DATA CHARACTERISTICS:
        - 200 sample construction projects
        - Realistic project attributes (budget, team size, complexity)
        - Calculated target variable based on business logic
        - Includes both numeric and categorical features
        
        SAMPLE DATA FIELDS:
        - Project_Name: Unique project identifiers
        - Budget: Project budget range ($50K - $500K)
        - Team_Size: Number of team members (3-15)
        - Complexity: Low/Medium/High complexity levels
        - Duration_Days: Project duration (30-365 days)
        - Technology: Technology stack used
        - Priority: Project priority level
        - Estimate_Completion: Target variable (calculated realistically)
        """
        self.log("\\nCreating sample data file...")
        
        try:
            # Import required libraries (should be installed by now)
            import pandas as pd
            import numpy as np
            
            # GENERATE REALISTIC PROJECT DATA
            np.random.seed(42)  # Reproducible random data
            n_samples = 200
            
            # CREATE SAMPLE PROJECT DATASET
            sample_data = pd.DataFrame({
                # PROJECT IDENTIFIERS
                'Project_Name': [f'Project_{i:03d}' for i in range(n_samples)],
                
                # NUMERIC PROJECT FEATURES
                'Budget': np.random.uniform(50000, 500000, n_samples),         # $50K - $500K
                'Team_Size': np.random.randint(3, 15, n_samples),             # 3-15 people
                'Duration_Days': np.random.randint(30, 365, n_samples),       # 1 month - 1 year
                
                # CATEGORICAL PROJECT FEATURES
                'Complexity': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                'Technology': np.random.choice(['Python', 'Java', 'JavaScript', 'C++'], n_samples),
                'Priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples),
                
                # TARGET VARIABLE (to be calculated)
                'Estimate_Completion': None
            })
            
            # CALCULATE REALISTIC TARGET VARIABLE
            # Business logic: how different factors affect project completion estimates
            
            # Complexity impact on estimates
            complexity_multiplier = sample_data['Complexity'].map({
                'Low': 0.8,      # Simple projects cost less
                'Medium': 1.0,   # Baseline
                'High': 1.3      # Complex projects cost more
            })
            
            # Priority impact on estimates
            priority_multiplier = sample_data['Priority'].map({
                'Low': 0.9,      # Lower priority = lower estimates
                'Medium': 1.0,   # Baseline
                'High': 1.1,     # Higher priority = higher estimates
                'Critical': 1.2  # Critical projects get premium estimates
            })
            
            # BASE ESTIMATE CALCULATION
            # Realistic business formula combining multiple factors
            base_estimate = (
                sample_data['Budget'] * 0.85 +                    # 85% of budget on average
                sample_data['Team_Size'] * 1000 +                 # $1000 per team member
                sample_data['Duration_Days'] * 100                # $100 per day
            )
            
            # FINAL ESTIMATE with business multipliers and realistic variation
            sample_data['Estimate_Completion'] = (
                base_estimate * complexity_multiplier * priority_multiplier +
                np.random.normal(0, base_estimate * 0.1)          # Add 10% random variation
            ).round(2)
            
            # SAVE SAMPLE DATA
            sample_data.to_csv('data/sample_project_data.csv', index=False)
            self.log("✅ Sample data created: data/sample_project_data.csv")
            
        except Exception as e:
            self.log(f"❌ Failed to create sample data: {e}")
    
    def create_config_file(self):
        """
        CONFIGURATION FILE CREATION: Set up application defaults
        
        BUSINESS PURPOSE:
        - Provides sensible default settings for immediate use
        - Allows easy customization without code changes
        - Documents all configurable options
        - Supports different deployment environments
        
        CONFIGURATION SECTIONS:
        1. App Settings: General application behavior
        2. Model Defaults: Machine learning algorithm settings
        3. Paths: Directory locations for organized file storage
        
        SETTINGS EXPLANATION:
        - default_train_size (0.8): Use 80% of data for training, 20% for testing
        - remove_outliers (True): Automatically clean unusual data points
        - max_display_rows (1000): Limit table display for performance
        """
        self.log("\\nCreating configuration file...")
        
        # DEFINE APPLICATION CONFIGURATION
        config = {
            # GENERAL APPLICATION SETTINGS
            "app_settings": {
                "default_train_size": 0.8,      # 80% training, 20% testing split
                "remove_outliers": True,         # Automatically remove data outliers
                "max_display_rows": 1000         # Limit displayed rows for performance
            },
            
            # MACHINE LEARNING MODEL DEFAULTS
            "model_defaults": {
                "random_forest": {
                    "n_estimators": 100,         # Number of trees in forest
                    "max_depth": 10              # Maximum tree depth
                },
                "xgboost": {
                    "n_estimators": 100,         # Number of boosting rounds
                    "learning_rate": 0.1         # Learning rate for gradient boosting
                }
            },
            
            # FILE SYSTEM PATHS
            "paths": {
                "data_directory": "data",        # Input data location
                "models_directory": "models",    # Saved models location
                "outputs_directory": "outputs",  # Results output location
                "logs_directory": "logs"         # Application logs location
            }
        }
        
        try:
            # Save configuration as JSON file
            with open('app_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            self.log("✅ Configuration file created: app_config.json")
        except Exception as e:
            self.log(f"❌ Failed to create config file: {e}")
    
    # --------------------------------------------------------------------------
    # FUNCTIONALITY TESTING SUITE
    # --------------------------------------------------------------------------
    def run_tests(self):
        """
        COMPREHENSIVE FUNCTIONALITY TESTING: Validate complete system
        
        TESTING STRATEGY:
        Run critical functionality tests to ensure the installation
        actually works and users won't encounter failures when using the app
        
        TEST SUITE:
        1. Module Import Test: Can we import all required libraries?
        2. Data Operations Test: Basic pandas functionality working?
        3. Machine Learning Test: Can we train and predict with models?
        4. GUI Availability Test: Is the user interface available?
        
        BUSINESS VALUE:
        - Catches installation problems before users encounter them
        - Provides confidence that the system is ready for use
        - Identifies specific components that need attention
        - Reduces support calls and user frustration
        """
        self.log("\\n" + "="*50)
        self.log("RUNNING FUNCTIONALITY TESTS")
        self.log("="*50)
        
        tests_passed = 0
        total_tests = 4
        
        # TEST 1: Module Import Verification
        # Verify all required libraries can be imported successfully
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import sklearn
            import xgboost as xgb
            import scipy
            self.log("✅ Test 1: All required modules imported successfully")
            tests_passed += 1
        except Exception as e:
            self.log(f"❌ Test 1: Module import failed: {e}")
        
        # TEST 2: Basic Data Operations
        # Test fundamental pandas operations that the app relies on
        try:
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            assert len(df) == 3
            assert list(df.columns) == ['a', 'b']
            self.log("✅ Test 2: Basic pandas operations working")
            tests_passed += 1
        except Exception as e:
            self.log(f"❌ Test 2: Pandas operations failed: {e}")
        
        # TEST 3: Machine Learning Operations
        # Test that we can train models and make predictions
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            import xgboost as xgb
            
            # Create simple test data
            X = np.random.randn(50, 3)   # 50 samples, 3 features
            y = np.random.randn(50)      # 50 target values
            
            # Test each model type used in the application
            lr = LinearRegression().fit(X, y)
            rf = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
            xgb_model = xgb.XGBRegressor(n_estimators=10, random_state=42).fit(X, y)
            
            # Test predictions (the core functionality)
            lr.predict(X[:5])
            rf.predict(X[:5])
            xgb_model.predict(X[:5])
            
            self.log("✅ Test 3: Machine learning models working")
            tests_passed += 1
        except Exception as e:
            self.log(f"❌ Test 3: ML operations failed: {e}")
        
        # TEST 4: GUI Availability Test
        # Verify that the graphical user interface can be created
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the test window
            root.destroy()   # Clean up
            self.log("✅ Test 4: GUI (tkinter) is available")
            tests_passed += 1
        except Exception as e:
            self.log(f"❌ Test 4: GUI test failed: {e}")
        
        # TEST RESULTS SUMMARY
        self.log(f"\\nTest Results: {tests_passed}/{total_tests} tests passed")
        return tests_passed == total_tests
    
    # --------------------------------------------------------------------------
    # INSTALLATION REPORTING
    # --------------------------------------------------------------------------
    def generate_installation_report(self):
        """
        COMPREHENSIVE INSTALLATION REPORT: Document complete installation process
        
        BUSINESS PURPOSE:
        - Provides audit trail for IT compliance
        - Helps troubleshoot future issues
        - Documents successful installation for reference
        - Guides users on next steps and usage
        
        REPORT CONTENTS:
        - Installation environment details
        - Complete log of all installation steps
        - Success/failure status with specific details
        - Next steps and usage instructions
        - Troubleshooting guidance for common issues
        """
        report_path = "installation_report.txt"
        
        # GENERATE COMPREHENSIVE REPORT
        report_content = f"""
PROJECT ESTIMATION ML APP - INSTALLATION REPORT
================================================
Installation Date: {Path().cwd()}
Platform: {self.platform}
Python Version: {sys.version}

INSTALLATION LOG:
{chr(10).join(self.installation_log)}

INSTALLATION STATUS: {'SUCCESS' if not self.failed_packages else 'PARTIAL'}

Next Steps:
1. Run the application: python project_estimation_app.py
2. Load sample data from: data/sample_project_data.csv
3. Check the README.md for detailed usage instructions

If you encounter issues:
- Check that all dependencies are installed: pip list
- Verify tkinter is available for GUI functionality
- Review the installation log above for specific errors

For support, review the error messages and ensure all requirements are met.
"""
        
        try:
            with open(report_path, 'w') as f:
                f.write(report_content)
            self.log(f"\\n📋 Installation report saved to: {report_path}")
        except Exception as e:
            self.log(f"❌ Failed to save installation report: {e}")
    
    # --------------------------------------------------------------------------
    # MASTER INSTALLATION ORCHESTRATOR
    # --------------------------------------------------------------------------
    def run_full_installation(self):
        """
        COMPLETE INSTALLATION PROCESS: Execute all installation steps
        
        INSTALLATION WORKFLOW:
        1. System Prerequisite Checks
        2. Package Installation and Verification
        3. Directory Structure Creation
        4. Sample Data and Configuration Setup
        5. Comprehensive Functionality Testing
        6. Installation Report Generation
        7. Final Status Determination
        
        BUSINESS OUTCOME:
        - Fully functional application ready for immediate use
        - Complete confidence in system reliability
        - Professional installation experience
        - Clear guidance for next steps
        
        Returns: True if completely successful, False if any issues
        """
        self.log("🚀 Starting Project Estimation ML App Installation")
        self.log("="*60)
        
        # PHASE 1: Check Prerequisites
        if not self.check_python_version():
            return False
        
        if not self.check_pip():
            return False
        
        # PHASE 2: Upgrade Package Manager
        self.upgrade_pip()
        
        # PHASE 3: Install All Required Packages
        packages_installed = self.install_requirements()
        
        # PHASE 4: Set Up Project Structure
        self.create_directory_structure()
        
        # PHASE 5: Create Initial Data and Configuration
        self.create_sample_data()
        self.create_config_file()
        
        # PHASE 6: Run Comprehensive Tests
        tests_passed = self.run_tests()
        
        # PHASE 7: Generate Installation Documentation
        self.generate_installation_report()
        
        # PHASE 8: Final Status and User Guidance
        self.log("\\n" + "="*60)
        if packages_installed and tests_passed:
            self.log("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
            self.log("\\nTo start the application, run:")
            self.log("   python project_estimation_app.py")
        else:
            self.log("⚠️ INSTALLATION COMPLETED WITH ISSUES")
            self.log("Please review the installation report for details.")
        
        self.log("="*60)
        
        return packages_installed and tests_passed


# ==============================================================================
# USER INTERACTION AND MAIN EXECUTION
# ==============================================================================
def main():
    """
    MAIN INSTALLATION INTERFACE: Handle user interaction and execute installation
    
    USER EXPERIENCE:
    - Clear welcome message explaining what will happen
    - User confirmation before beginning installation
    - Real-time progress feedback during installation
    - Clear success/failure status and next steps
    
    BUSINESS BENEFITS:
    - Professional installation experience
    - User control over when installation occurs
    - Clear feedback and status reporting
    - Appropriate guidance for next steps
    """
    installer = InstallationManager()
    
    print("Welcome to Project Estimation ML App Installation!")
    print("This will install all required dependencies and set up the application.")
    
    # REQUEST USER CONFIRMATION
    # Don't start installation without explicit user consent
    response = input("\\nProceed with installation? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # USER CONFIRMED - Run complete installation
        success = installer.run_full_installation()
        
        if success:
            print("\\n✅ Installation completed successfully!")
            print("You can now run the application with: python project_estimation_app.py")
        else:
            print("\\n❌ Installation completed with some issues.")
            print("Please check the installation report for details.")
            
        return success
    else:
        # USER DECLINED - Exit gracefully
        print("Installation cancelled by user.")
        return False


# ==============================================================================
# SCRIPT EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    """
    SCRIPT EXECUTION: Run installation when script is executed directly
    
    EXECUTION FLOW:
    1. Run main installation function
    2. Set appropriate exit code for automation systems
    3. Exit code 0 = success, exit code 1 = failure
    
    AUTOMATION SUPPORT:
    The exit code allows this script to be used in automated deployment
    systems where success/failure status needs to be programmatically detected
    """
    success = main()
    sys.exit(0 if success else 1)


# ==============================================================================
# INSTALLATION TROUBLESHOOTING GUIDE
# ==============================================================================
"""
COMMON INSTALLATION ISSUES AND SOLUTIONS:

1. PYTHON VERSION TOO OLD:
   Problem: Python 3.6 or earlier
   Solution: Install Python 3.7+ from python.org

2. PIP NOT AVAILABLE:
   Problem: pip command not found
   Solution: Reinstall Python with pip included, or install pip separately

3. PERMISSION ERRORS:
   Problem: Cannot install packages system-wide
   Solution: Use virtual environment or --user flag

4. NETWORK CONNECTIVITY:
   Problem: Cannot download packages
   Solution: Check internet connection, configure proxy if needed

5. TKINTER MISSING:
   Problem: GUI cannot start
   Solution: Install tkinter package for your operating system

6. PACKAGE CONFLICTS:
   Problem: Incompatible package versions
   Solution: Use virtual environment to isolate dependencies

7. MEMORY ERRORS DURING INSTALLATION:
   Problem: Out of memory while installing large packages
   Solution: Close other applications, upgrade system memory

8. ANTIVIRUS INTERFERENCE:
   Problem: Antivirus blocking package downloads
   Solution: Temporarily disable antivirus or add Python to whitelist

SYSTEM REQUIREMENTS:
- Python 3.7 or higher
- 2GB+ available memory
- 1GB+ disk space
- Internet connection for package downloads
- Admin privileges may be required for system-wide installation

DEPLOYMENT CHECKLIST:
□ Python 3.7+ installed
□ pip package manager available
□ Internet connectivity confirmed
□ Sufficient disk space available
□ User permissions adequate
□ Antivirus configured to allow Python
□ All prerequisites met

POST-INSTALLATION VERIFICATION:
1. Check that all packages import correctly
2. Verify sample data was created
3. Test basic GUI functionality
4. Run a simple prediction to ensure ML pipeline works
5. Check that all directories were created
6. Verify configuration file exists and is valid

MAINTENANCE RECOMMENDATIONS:
- Keep Python and packages updated
- Regular backup of trained models and data
- Monitor disk space usage in data directories
- Review logs periodically for issues
- Update configuration as business needs change

UNINSTALLATION:
To remove the application:
1. Delete the application directory
2. Remove installed packages if no longer needed:
   pip uninstall pandas numpy matplotlib seaborn scikit-learn xgboost scipy joblib openpyxl
3. Clean up any remaining configuration files

SUPPORT AND DOCUMENTATION:
- Installation report provides detailed logs
- Configuration file documents all settings
- Sample data demonstrates expected format
- Application includes built-in help and documentation
- Error messages provide specific guidance for resolution
"""