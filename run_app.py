#!/usr/bin/env python3
# ==============================================================================
# APPLICATION LAUNCHER MODULE
# ==============================================================================
#
# PURPOSE: Smart application startup system with comprehensive environment validation
#
# WHAT THIS LAUNCHER DOES:
# - Validates the complete system environment before starting the application
# - Checks for all required dependencies and provides helpful error messages
# - Creates necessary directories and sets up logging
# - Provides user-friendly error handling and recovery options
# - Ensures the application starts reliably across different environments
#
# BUSINESS BENEFITS:
# - Prevents frustrating "application won't start" scenarios
# - Provides clear guidance when problems occur
# - Reduces support calls by catching issues early
# - Creates professional user experience from first launch
# - Automatically handles common setup issues
#
# TARGET SCENARIOS:
# - First-time application startup after installation
# - Troubleshooting when the main application won't start
# - Validating environment after system changes
# - Professional deployment in enterprise environments
#
# REAL-WORLD ANALOGY:
# Like a pre-flight checklist for an airplane - systematically verifies
# that all systems are ready before takeoff, preventing failures during
# the actual mission
# ==============================================================================

"""
Project Estimation ML App Launcher
Comprehensive launcher with environment checks, error handling, and user guidance
"""

# STEP 1: IMPORT SYSTEM LIBRARIES
# These provide access to system functions, package management, and user interface
import sys                    # Python system information and control
import os                     # Operating system interface
import importlib.util         # Dynamic module importing for dependency testing
import subprocess            # Execute external commands (like running install script)
from pathlib import Path     # Modern file and directory handling
import tkinter as tk         # GUI framework for user interface
from tkinter import messagebox  # GUI dialog boxes for user communication
import logging               # Comprehensive logging system
from datetime import datetime    # Date and time handling for logs


# ==============================================================================
# MAIN APPLICATION LAUNCHER CLASS
# ==============================================================================
class AppLauncher:
    """
    COMPREHENSIVE APPLICATION LAUNCHER
    
    WHAT IT MANAGES:
    - Complete environment validation before application startup
    - Dependency checking with helpful error messages
    - Directory structure creation and maintenance
    - Professional logging and error tracking
    - User-friendly error dialogs and recovery options
    - Graceful handling of missing dependencies
    
    BUSINESS VALUE:
    - Eliminates "it doesn't work" support calls
    - Provides professional startup experience
    - Catches problems before they affect users
    - Guides users to solutions when issues occur
    - Creates audit trail of startup attempts
    
    LAUNCH PROCESS:
    1. Check Python version compatibility
    2. Verify all required dependencies are installed
    3. Validate application files are present
    4. Create necessary directory structure
    5. Set up logging and error handling
    6. Launch main application with monitoring
    """
    
    def __init__(self):
        """
        INITIALIZATION: Set up dependency requirements and logging
        
        DEPENDENCY STRATEGY:
        - required_packages: Must have these or application cannot function
        - optional_packages: Nice to have, but application can work without them
        - Clear mapping between import names and pip install names
        """
        
        # REQUIRED DEPENDENCIES: Application cannot function without these
        self.required_packages = {
            'pandas': 'pandas',              # Data manipulation and analysis
            'numpy': 'numpy',               # Numerical computing foundation
            'matplotlib': 'matplotlib',     # Basic plotting and visualization
            'seaborn': 'seaborn',          # Statistical visualization
            'sklearn': 'scikit-learn',     # Machine learning algorithms
            'xgboost': 'xgboost',          # Advanced gradient boosting
            'scipy': 'scipy'               # Scientific computing utilities
        }
        
        # OPTIONAL DEPENDENCIES: Enhance functionality but not critical
        self.optional_packages = {
            'joblib': 'joblib',            # Model persistence and parallel processing
            'openpyxl': 'openpyxl'         # Excel file reading/writing
        }
        
        # TRACKING VARIABLES
        self.missing_packages = []          # Track packages that need installation
        self.setup_logging()               # Initialize logging system
    
    # --------------------------------------------------------------------------
    # LOGGING SYSTEM SETUP
    # --------------------------------------------------------------------------
    def setup_logging(self):
        """
        PROFESSIONAL LOGGING SETUP: Create comprehensive logging system
        
        LOGGING STRATEGY:
        - File logging: Permanent record for troubleshooting
        - Console logging: Real-time feedback to users
        - Timestamped log files: Easy identification and organization
        - Both INFO and ERROR levels: Complete activity tracking
        
        BUSINESS PURPOSE:
        - Troubleshoot startup issues effectively
        - Audit trail for IT compliance
        - Performance monitoring and optimization
        - User support and debugging assistance
        """
        
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file for this launch attempt
        log_file = log_dir / f'app_launch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # Configure logging with both file and console output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),  # Save to file for permanent record
                logging.StreamHandler()         # Display to console for immediate feedback
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    # --------------------------------------------------------------------------
    # SYSTEM COMPATIBILITY VALIDATION
    # --------------------------------------------------------------------------
    def check_python_version(self) -> bool:
        """
        PYTHON VERSION COMPATIBILITY CHECK
        
        BUSINESS IMPORTANCE:
        Modern machine learning libraries require recent Python versions.
        Using incompatible versions leads to cryptic import errors and
        frustrated users who can't figure out why nothing works.
        
        VERSION REQUIREMENTS:
        - Python 3.7+: Required for modern type hints and library compatibility
        - Older versions lack security updates and modern features
        - Many ML libraries drop support for older Python versions
        
        Returns: True if compatible, False if upgrade needed
        """
        min_version = (3, 7)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            self.logger.error(f"Python {min_version[0]}.{min_version[1]}+ required, "
                            f"found {current_version[0]}.{current_version[1]}")
            return False
        
        self.logger.info(f"Python version check passed: {sys.version}")
        return True
    
    def check_package(self, package_name: str, install_name: str) -> bool:
        """
        INDIVIDUAL PACKAGE AVAILABILITY CHECK
        
        TESTING STRATEGY:
        - Attempt to import the package
        - Handle special cases (like sklearn vs scikit-learn naming)
        - Provide helpful installation guidance if missing
        
        BUSINESS VALUE:
        - Identifies specific missing dependencies
        - Provides exact installation commands
        - Prevents confusing import errors during application use
        
        Parameters:
        - package_name: Name used for importing (e.g., 'sklearn')
        - install_name: Name used for pip install (e.g., 'scikit-learn')
        
        Returns: True if package is available, False if missing
        """
        try:
            # SPECIAL CASE HANDLING
            if package_name == 'sklearn':
                # scikit-learn imports as 'sklearn'
                import sklearn
            else:
                # Standard import mechanism
                importlib.import_module(package_name)
            return True
        except ImportError:
            # Package is missing - log helpful installation guidance
            self.logger.warning(f"Missing package: {package_name} (install with: pip install {install_name})")
            return False
    
    def check_tkinter(self) -> bool:
        """
        GUI FRAMEWORK AVAILABILITY CHECK
        
        BUSINESS IMPORTANCE:
        tkinter provides the graphical user interface. Without it, the application
        can only run in command-line mode, which defeats the purpose of a
        user-friendly ML application.
        
        TESTING APPROACH:
        - Import tkinter module
        - Create and destroy a test window to verify functionality
        - Handle various failure modes gracefully
        
        COMMON ISSUES:
        - tkinter missing on some Linux distributions
        - Display issues in remote/headless environments
        - Permission problems in containerized environments
        """
        try:
            import tkinter as tk
            # Test basic tkinter functionality by creating a window
            root = tk.Tk()
            root.withdraw()  # Hide the test window
            root.destroy()   # Clean up
            return True
        except ImportError:
            self.logger.error("tkinter not available - GUI will not work")
            return False
        except Exception as e:
            self.logger.error(f"tkinter test failed: {e}")
            return False
    
    def check_all_dependencies(self) -> bool:
        """
        COMPREHENSIVE DEPENDENCY VALIDATION
        
        VALIDATION PROCESS:
        1. Check all required packages (critical for functionality)
        2. Check optional packages (nice to have, warn if missing)
        3. Verify GUI framework availability
        4. Compile list of missing dependencies for user guidance
        
        BUSINESS OUTCOME:
        - Clear pass/fail status for application readiness
        - Specific list of missing packages for easy resolution
        - Distinction between critical and optional dependencies
        
        Returns: True if all critical dependencies available, False otherwise
        """
        self.logger.info("Checking dependencies...")
        
        all_good = True
        
        # CHECK REQUIRED PACKAGES (application cannot function without these)
        for package, install_name in self.required_packages.items():
            if not self.check_package(package, install_name):
                self.missing_packages.append(install_name)
                all_good = False
        
        # CHECK OPTIONAL PACKAGES (warn but don't fail)
        for package, install_name in self.optional_packages.items():
            if not self.check_package(package, install_name):
                self.logger.warning(f"Optional package missing: {install_name}")
        
        # CHECK GUI FRAMEWORK (critical for user interface)
        if not self.check_tkinter():
            all_good = False
        
        return all_good
    
    # --------------------------------------------------------------------------
    # APPLICATION FILE VALIDATION
    # --------------------------------------------------------------------------
    def check_app_files(self) -> bool:
        """
        APPLICATION FILE EXISTENCE CHECK
        
        BUSINESS IMPORTANCE:
        Verifies that the core application files are present before attempting
        to launch. Prevents confusing "module not found" errors that occur
        when users try to run incomplete installations.
        
        REQUIRED FILES:
        - project_estimation_app.py: Main application module
        - config.py: Configuration and settings
        
        Returns: True if all required files exist, False otherwise
        """
        required_files = [
            'project_estimation_app.py',    # Main application module
            'config.py'                     # Configuration settings
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.logger.error(f"Missing required files: {', '.join(missing_files)}")
            return False
        
        return True
    
    def create_directories(self):
        """
        DIRECTORY STRUCTURE CREATION
        
        BUSINESS PURPOSE:
        Ensures all necessary directories exist for the application to store
        data, models, outputs, and logs. Creates professional organization
        and prevents file system errors during operation.
        
        DIRECTORY STRUCTURE:
        - data/: Input datasets and sample files
        - models/: Trained machine learning models
        - outputs/: Prediction results and reports
        - logs/: Application logs and error tracking
        - exports/: User-exported files and results
        """
        directories = ['data', 'models', 'outputs', 'logs', 'exports']
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            self.logger.info(f"Directory ready: {directory}/")
    
    # --------------------------------------------------------------------------
    # USER COMMUNICATION SYSTEM
    # --------------------------------------------------------------------------
    def show_error_dialog(self, title: str, message: str):
        """
        USER-FRIENDLY ERROR COMMUNICATION
        
        COMMUNICATION STRATEGY:
        - Try to show GUI dialog for professional appearance
        - Fall back to console output if GUI unavailable
        - Provide clear, actionable error messages
        
        BUSINESS VALUE:
        - Professional error handling improves user experience
        - Clear guidance reduces support calls
        - Consistent error reporting across different failure modes
        """
        try:
            # Attempt to show professional GUI dialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showerror(title, message)
            root.destroy()
        except:
            # Fallback to console output if GUI unavailable
            print(f"ERROR - {title}: {message}")
    
    def show_missing_dependencies_help(self):
        """
        DEPENDENCY INSTALLATION GUIDANCE
        
        HELP STRATEGY:
        - Show specific list of missing packages
        - Provide exact installation commands
        - Offer multiple resolution options
        - Guide users to installation script if available
        
        USER EXPERIENCE:
        - Clear list of what's missing
        - Copy-paste ready installation commands
        - Professional error dialog with guidance
        - Console output as backup communication method
        """
        if not self.missing_packages:
            return
        
        # Create user-friendly installation command
        install_command = f"pip install {' '.join(self.missing_packages)}"
        
        # Comprehensive help message
        message = f"""Missing Required Dependencies:
{chr(10).join(f'• {pkg}' for pkg in self.missing_packages)}

To install missing packages, run:
{install_command}

Or run the installation script:
python install.py"""
        
        self.logger.error("Missing dependencies detected")
        self.show_error_dialog("Missing Dependencies", message)
        
        # Also show in console for copy-paste convenience
        print("\\n" + "="*60)
        print("MISSING DEPENDENCIES DETECTED")
        print("="*60)
        print(message)
        print("="*60)
    
    # --------------------------------------------------------------------------
    # APPLICATION LAUNCH SYSTEM
    # --------------------------------------------------------------------------
    def launch_application(self) -> bool:
        """
        MAIN APPLICATION LAUNCH WITH ERROR HANDLING
        
        LAUNCH PROCESS:
        1. Import the main application module
        2. Create the GUI root window
        3. Initialize the application class
        4. Set up comprehensive error handling
        5. Start the GUI event loop
        6. Monitor for exceptions and handle gracefully
        
        ERROR HANDLING STRATEGY:
        - Catch import errors (missing modules)
        - Handle GUI creation failures
        - Set up runtime exception handling
        - Provide user-friendly error messages
        - Log all errors for troubleshooting
        
        BUSINESS BENEFITS:
        - Professional application startup experience
        - Clear error messages when problems occur
        - Comprehensive logging for support purposes
        - Graceful handling of unexpected issues
        """
        try:
            self.logger.info("Starting Project Estimation ML App...")
            
            # STEP 1: Import main application module
            from project_estimation_app import ProjectEstimationApp
            
            # STEP 2: Create GUI framework
            root = tk.Tk()
            app = ProjectEstimationApp(root)
            
            # STEP 3: Set up runtime exception handling
            def handle_exception(exc_type, exc_value, exc_traceback):
                """
                RUNTIME EXCEPTION HANDLER
                
                Catches unexpected errors during application use and presents
                them professionally to users while logging for troubleshooting
                """
                # Allow keyboard interrupts to work normally
                if issubclass(exc_type, KeyboardInterrupt):
                    sys.__excepthook__(exc_type, exc_value, exc_traceback)
                    return
                
                # Handle unexpected errors gracefully
                error_msg = f"An unexpected error occurred:\\n{exc_type.__name__}: {exc_value}"
                self.logger.error(error_msg, exc_info=True)
                messagebox.showerror("Application Error", error_msg)
            
            # Install the exception handler
            sys.excepthook = handle_exception
            
            # STEP 4: Start the application
            self.logger.info("Application GUI started successfully")
            root.mainloop()  # Begin GUI event loop
            
            return True
            
        except ImportError as e:
            # Handle missing application modules
            error_msg = f"Failed to import main application: {e}"
            self.logger.error(error_msg)
            self.show_error_dialog("Import Error", error_msg)
            return False
        
        except Exception as e:
            # Handle any other startup failures
            error_msg = f"Failed to start application: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.show_error_dialog("Launch Error", error_msg)
            return False
    
    # --------------------------------------------------------------------------
    # COMPREHENSIVE ENVIRONMENT VALIDATION
    # --------------------------------------------------------------------------
    def run_installation_check(self) -> bool:
        """
        COMPLETE ENVIRONMENT VALIDATION PROCESS
        
        VALIDATION CHECKLIST:
        ✓ Python version compatibility
        ✓ All required dependencies installed
        ✓ Application files present
        ✓ Directory structure ready
        ✓ Logging system operational
        
        BUSINESS OUTCOME:
        - High confidence that application will start successfully
        - Clear identification of any blocking issues
        - Professional status reporting to users
        - Comprehensive logging for troubleshooting
        
        Returns: True if environment is ready, False if issues found
        """
        print("🔍 Checking Project Estimation ML App Environment...")
        print("="*60)
        
        # CHECK 1: Python Version Compatibility
        if not self.check_python_version():
            return False
        
        # CHECK 2: Dependency Availability
        if not self.check_all_dependencies():
            self.show_missing_dependencies_help()
            return False
        
        # CHECK 3: Application Files Present
        if not self.check_app_files():
            return False
        
        # CHECK 4: Directory Structure Ready
        self.create_directories()
        
        print("✅ All checks passed!")
        return True
    
    # --------------------------------------------------------------------------
    # MASTER LAUNCH ORCHESTRATOR
    # --------------------------------------------------------------------------
    def run(self) -> bool:
        """
        MAIN LAUNCHER FUNCTION: Complete startup process
        
        STARTUP WORKFLOW:
        1. Run comprehensive environment validation
        2. Launch main application if validation passes
        3. Handle user interruptions gracefully
        4. Log all activities for troubleshooting
        
        ERROR HANDLING:
        - Keyboard interrupts (Ctrl+C) handled gracefully
        - All exceptions logged with full details
        - Clear status reporting to users
        
        BUSINESS OUTCOME:
        - Reliable application startup across environments
        - Professional user experience
        - Complete audit trail of startup attempts
        
        Returns: True if successful startup, False if any issues
        """
        try:
            # PHASE 1: Environment Validation
            if not self.run_installation_check():
                return False
            
            # PHASE 2: Application Launch
            return self.launch_application()
            
        except KeyboardInterrupt:
            # Handle user cancellation gracefully
            print("\\n👋 Application startup cancelled by user")
            return False
        
        except Exception as e:
            # Handle unexpected launcher failures
            error_msg = f"Launcher failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            print(f"❌ {error_msg}")
            return False


# ==============================================================================
# USER INTERFACE AND MESSAGING
# ==============================================================================
def show_welcome_message():
    """
    PROFESSIONAL WELCOME MESSAGE
    
    BUSINESS PURPOSE:
    - Creates professional first impression
    - Sets user expectations about application capabilities
    - Builds confidence in the software quality
    - Provides clear feature overview
    
    MESSAGE STRATEGY:
    - Highlight key business benefits
    - List major features concisely
    - Create excitement about capabilities
    - Set professional tone for the application
    """
    print("""
🚀 PROJECT ESTIMATION ML APP
============================
Advanced Machine Learning for Project Completion Estimation

Features:
• Multiple regression models (Linear, Random Forest, XGBoost)
• Intelligent data type detection
• Comprehensive visualizations
• Batch and single predictions
• Feature importance analysis
• Outlier detection and removal

Starting application...
""")


# ==============================================================================
# MAIN EXECUTION AND USER INTERACTION
# ==============================================================================
def main():
    """
    MAIN FUNCTION: User interaction and launcher execution
    
    USER EXPERIENCE FLOW:
    1. Show professional welcome message
    2. Execute comprehensive environment validation
    3. Launch application if ready
    4. Provide helpful guidance if issues occur
    5. Offer automatic problem resolution when possible
    
    RECOVERY OPTIONS:
    - Clear error messages with specific guidance
    - Automatic installation script execution
    - Manual installation commands
    - Links to troubleshooting resources
    
    BUSINESS BENEFITS:
    - Professional user experience from first contact
    - Self-service problem resolution reduces support load
    - Clear guidance prevents user frustration
    - Comprehensive error reporting aids troubleshooting
    """
    show_welcome_message()
    
    # Execute main launcher
    launcher = AppLauncher()
    success = launcher.run()
    
    if not success:
        # APPLICATION LAUNCH FAILED - Provide helpful recovery guidance
        print("\\n❌ Failed to start application")
        print("For help, check:")
        print("• Installation report: installation_report.txt")
        print("• Application logs: logs/")
        print("• README.md for troubleshooting")
        
        # OFFER AUTOMATIC PROBLEM RESOLUTION
        try:
            response = input("\\nWould you like to run the installation script? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("Running installation script...")
                try:
                    # Execute installation script automatically
                    subprocess.run([sys.executable, "install.py"], check=True)
                    print("Installation completed. Please try running the app again.")
                except subprocess.CalledProcessError:
                    print("Installation script failed. Please install dependencies manually.")
                except FileNotFoundError:
                    print("Installation script not found. Please install dependencies manually:")
                    print("pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy")
        except KeyboardInterrupt:
            print("\\nExiting...")
        
        return False
    
    # APPLICATION CLOSED SUCCESSFULLY
    print("\\n👋 Application closed successfully")
    return True


# ==============================================================================
# SCRIPT EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    """
    SCRIPT EXECUTION: Run launcher when script is executed directly
    
    EXECUTION FLOW:
    1. Run main launcher function
    2. Set appropriate exit code for automation systems
    3. Exit code 0 = success, exit code 1 = failure
    
    AUTOMATION SUPPORT:
    The exit code allows this script to be used in automated deployment
    systems where success/failure status needs to be programmatically detected
    """
    success = main()
    sys.exit(0 if success else 1)


# ==============================================================================
# LAUNCHER TROUBLESHOOTING GUIDE
# ==============================================================================
"""
COMMON LAUNCHER ISSUES AND SOLUTIONS:

1. PYTHON VERSION TOO OLD:
   Problem: "Python 3.7+ required" error
   Solution: Install Python 3.7+ from python.org or system package manager

2. MISSING DEPENDENCIES:
   Problem: "Missing package" errors
   Solution: Run automatic installation or use provided pip commands

3. TKINTER NOT AVAILABLE:
   Problem: "GUI will not work" error
   Solution: Install tkinter package for your operating system
   - Ubuntu/Debian: sudo apt-get install python3-tk
   - CentOS/RHEL: sudo yum install tkinter
   - macOS: Usually included, or use Homebrew

4. APPLICATION FILES MISSING:
   Problem: "Missing required files" error
   Solution: Ensure complete application download/extraction

5. PERMISSION ERRORS:
   Problem: Cannot create directories or log files
   Solution: Run with appropriate permissions or choose different location

6. IMPORT ERRORS:
   Problem: Application modules cannot be imported
   Solution: Verify all files are in same directory, check file corruption

ENVIRONMENT VALIDATION CHECKLIST:
□ Python 3.7+ installed and accessible
□ All required packages available (pandas, numpy, matplotlib, seaborn, sklearn, xgboost, scipy)
□ tkinter GUI framework available
□ Application files present (project_estimation_app.py, config.py)
□ Write permissions for logs and directories
□ Sufficient disk space for operation

LOGGING AND DIAGNOSTICS:
- Launch logs: logs/app_launch_*.log
- Application logs: logs/ directory
- Installation report: installation_report.txt
- Console output provides real-time feedback

PROFESSIONAL DEPLOYMENT:
- Use virtual environments for isolation
- Validate all dependencies before deployment
- Test launcher in target environment
- Document any environment-specific requirements
- Set up monitoring for launch success/failure rates

MAINTENANCE RECOMMENDATIONS:
- Regularly update dependencies for security and features
- Monitor log files for recurring issues
- Update minimum Python version as needed
- Test launcher after system updates
- Document any environment-specific configurations
"""