# ==============================================================================
# BATCH PROCESSOR MODULE
# ==============================================================================
# 
# PURPOSE: This module handles large datasets that are too big to fit in memory
# Think of it like processing a massive Excel file one page at a time instead 
# of trying to load the entire thing at once
#
# BUSINESS VALUE: 
# - Prevents computer crashes from memory overload
# - Processes millions of rows efficiently 
# - Tracks progress so users know how long operations will take
# - Trains multiple AI models simultaneously to save time
# ==============================================================================

# STEP 1: IMPORT NECESSARY LIBRARIES
# These are like importing different toolboxes for specific jobs
import pandas as pd              # Data manipulation (like advanced Excel)
import numpy as np              # Mathematical operations and arrays
from typing import Dict, List, Optional, Generator, Callable  # Type hints for code clarity
import os                       # File and folder operations
from concurrent.futures import ThreadPoolExecutor, as_completed  # Parallel processing
import threading               # Thread-safe operations
from datetime import datetime  # Date and time handling
import json                   # JSON file handling
import pickle                 # Python object serialization
from config import Config     # Application configuration settings


# ==============================================================================
# MAIN BATCH PROCESSOR CLASS
# ==============================================================================
# This is the main "engine" that handles all large-scale data processing
class BatchProcessor:
    """
    WHAT IT DOES: Processes large datasets efficiently without crashing your computer
    
    WHY WE NEED IT: 
    - Normal data processing loads everything into memory at once
    - Large files (millions of rows) can crash the system
    - This processes data in small "chunks" or "batches"
    
    REAL-WORLD ANALOGY: 
    Like reading a 1000-page book one chapter at a time instead of 
    trying to memorize the whole book at once
    """
    
    def __init__(self, chunk_size: int = None):
        """
        INITIALIZATION: Sets up the batch processor with default settings
        
        Parameters:
        - chunk_size: How many rows to process at once (like page size in a book)
        """
        # Set chunk size from config or use provided value
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        
        # Initialize tracking variables
        self.progress_callback = None    # Function to report progress updates
        self.results_cache = {}          # Store results temporarily  
        self.processing_stats = {}       # Keep track of performance metrics
    
    # --------------------------------------------------------------------------
    # PROGRESS TRACKING SETUP
    # --------------------------------------------------------------------------
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """
        WHAT IT DOES: Sets up progress reporting (like a progress bar)
        
        WHY IT'S USEFUL: 
        - Users can see how long processing will take
        - Prevents the "Is it broken?" question during long operations
        
        Parameters:
        - callback: A function that gets called with progress updates
        """
        self.progress_callback = callback
    
    # --------------------------------------------------------------------------
    # MAIN PROCESSING ENGINE
    # --------------------------------------------------------------------------
    def process_large_dataset(self, 
                            file_path: str,
                            processing_func: Callable,
                            output_path: str = None) -> Dict:
        """
        CORE FUNCTIONALITY: The main engine that processes large files efficiently
        
        STEP-BY-STEP PROCESS:
        1. Opens the file and counts total rows (for progress tracking)
        2. Reads the file in small chunks (like reading page by page)
        3. Processes each chunk using the provided function
        4. Combines all processed chunks back together
        5. Saves the final result
        6. Returns performance statistics
        
        BUSINESS BENEFITS:
        - Handles files with millions of rows without crashes
        - Shows real-time progress to users
        - Provides detailed performance metrics
        - Memory-efficient processing
        
        Parameters:
        - file_path: Location of the input file
        - processing_func: What to do with each chunk of data
        - output_path: Where to save the final result
        
        Returns:
        - Dictionary containing processed data and performance stats
        """
        
        # SUBSTEP 1: Initialize tracking variables
        start_time = datetime.now()      # Record when we started
        processed_chunks = 0             # Count how many chunks we've completed
        total_rows = 0                   # Total number of rows in the file
        
        # SUBSTEP 2: Count total rows for progress tracking
        # This is like counting total pages in a book before reading
        try:
            if file_path.endswith('.csv'):
                # For CSV files: count lines and subtract header
                total_rows = sum(1 for _ in open(file_path)) - 1
            else:
                # For Excel files: need to load to get size, one case for excel binary files and another for the normal xlsx files
                if file_path.endswith('.xlsb'):
                    total_rows = len(pd.read_excel(file_path, engine='pyxlsb'))
                else:
                    total_rows = len(pd.read_excel(file_path))
        except Exception as e:
            print(f"Could not determine file size: {e}")
            total_rows = None
        
        # SUBSTEP 3: Initialize results container
        processed_data = []
        
        try:
            # SUBSTEP 4: Process the file chunk by chunk
            # This is the main processing loop - like reading a book chapter by chapter
            for chunk_num, chunk in enumerate(self._read_chunks(file_path)):
                
                # Update progress if we have a progress tracker
                if self.progress_callback and total_rows:
                    progress = min(100, (processed_chunks * self.chunk_size / total_rows) * 100)
                    self.progress_callback(int(progress), 100, f"Processing chunk {chunk_num + 1}")
                
                # Apply the processing function to this chunk
                processed_chunk = processing_func(chunk)
                processed_data.append(processed_chunk)
                processed_chunks += 1
                
                # MEMORY MANAGEMENT: Clean up to prevent memory buildup
                del chunk  # Delete the original chunk from memory
                if chunk_num % 10 == 0:  # Every 10 chunks, force garbage collection
                    import gc
                    gc.collect()
            
            # SUBSTEP 5: Combine all processed chunks into final result
            if processed_data:
                final_result = pd.concat(processed_data, ignore_index=True)
                
                # SUBSTEP 6: Save results if output path provided
                if output_path:
                    self._save_processed_data(final_result, output_path)
                
                # SUBSTEP 7: Calculate performance statistics
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Create detailed stats for reporting
                stats = {
                    'chunks_processed': processed_chunks,
                    'total_rows': len(final_result),
                    'processing_time': processing_time,
                    'rows_per_second': len(final_result) / processing_time if processing_time > 0 else 0,
                    'memory_efficient': True,
                    'output_path': output_path
                }
                
                # Store stats for later reporting
                self.processing_stats[file_path] = stats
                return {'data': final_result, 'stats': stats}
            
        except Exception as e:
            # If anything goes wrong, return error information
            return {'error': f'Batch processing failed: {str(e)}'}
        
        # If no data was processed, return error
        return {'error': 'No data processed'}
    
    # --------------------------------------------------------------------------
    # FILE READING HELPER FUNCTION
    # --------------------------------------------------------------------------
    def _read_chunks(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """
        HELPER FUNCTION: Reads large files in manageable pieces
        
        WHAT IT DOES:
        - Takes a large file and breaks it into smaller pieces
        - Works with both CSV and Excel files
        - Returns one piece at a time (Generator pattern)
        
        ANALOGY: Like tearing pages out of a phone book one section at a time
        instead of trying to carry the whole book
        
        Parameters:
        - file_path: Location of the file to read
        
        Yields:
        - One chunk (piece) of the file at a time
        """
        if file_path.endswith('.csv'):
            # For CSV files: use pandas built-in chunking
            chunk_reader = pd.read_csv(file_path, chunksize=self.chunk_size)
            for chunk in chunk_reader:
                yield chunk
                
        elif file_path.endswith(('.xlsx', '.xls', '.xlsb')):
            # For Excel files: read entire file then slice it
            # (Excel files can't be chunked during reading like CSV)
            if file_path.endswith('.xlsb'):
                df = pd.read_excel(file_path, engine='pyxlsb')
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            
            # Slice the dataframe into chunks
            for start in range(0, len(df), self.chunk_size):
                end = min(start + self.chunk_size, len(df))
                yield df.iloc[start:end].copy()
    
    # --------------------------------------------------------------------------
    # FILE SAVING HELPER FUNCTION
    # --------------------------------------------------------------------------
    def _save_processed_data(self, data: pd.DataFrame, output_path: str):
        """
        HELPER FUNCTION: Saves processed data in the correct format
        
        WHAT IT DOES:
        - Automatically detects desired format from file extension
        - Saves data in CSV, Excel, or Pickle format
        - Handles errors gracefully
        
        Parameters:
        - data: The processed dataframe to save
        - output_path: Where to save it and in what format
        """
        try:
            if output_path.endswith('.csv'):
                data.to_csv(output_path, index=False)
            elif output_path.endswith(('.xlsx', '.xls')):
                data.to_excel(output_path, index=False)
            elif output_path.endswith('.pickle'):
                data.to_pickle(output_path)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    # --------------------------------------------------------------------------
    # BATCH PREDICTION ENGINE
    # --------------------------------------------------------------------------
    def batch_predict(self, 
                     models: Dict,
                     data_path: str,
                     output_path: str,
                     preprocessing_func: Optional[Callable] = None) -> Dict:
        """
        SPECIALIZED PROCESSING: Uses trained AI models to make predictions on large datasets
        
        BUSINESS PURPOSE:
        - Apply trained machine learning models to new data
        - Handle datasets too large for normal prediction methods
        - Generate predictions efficiently without memory issues
        
        PROCESS OVERVIEW:
        1. Reads data in chunks
        2. Applies preprocessing if needed
        3. Runs predictions using multiple models
        4. Saves results with all predictions included
        
        Parameters:
        - models: Dictionary of trained machine learning models
        - data_path: File containing data to predict on
        - output_path: Where to save predictions
        - preprocessing_func: Optional data cleaning function
        
        Returns:
        - Dictionary with prediction results and statistics
        """
        
        def predict_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            """
            INNER FUNCTION: Processes one chunk of data for predictions
            
            STEPS:
            1. Copy the original data
            2. Apply preprocessing (data cleaning)
            3. Run each model to get predictions
            4. Add prediction columns to the data
            5. Return enhanced data with predictions
            """
            processed_chunk = chunk.copy()
            
            # STEP 1: Apply preprocessing if provided
            if preprocessing_func:
                processed_chunk = preprocessing_func(processed_chunk)
            
            # STEP 2: Make predictions with all models
            for model_name, model in models.items():
                try:
                    # Select only numeric columns for prediction
                    numeric_data = processed_chunk.select_dtypes(include=[np.number])
                    predictions = model.predict(numeric_data)
                    processed_chunk[f'{model_name}_prediction'] = predictions
                except Exception as e:
                    print(f"Prediction failed for {model_name}: {e}")
                    # If prediction fails, add NaN column
                    processed_chunk[f'{model_name}_prediction'] = np.nan
            
            return processed_chunk
        
        # Use the main processing engine with our prediction function
        return self.process_large_dataset(data_path, predict_chunk, output_path)
    
    # --------------------------------------------------------------------------
    # PARALLEL MODEL TRAINING
    # --------------------------------------------------------------------------
    def parallel_model_training(self, 
                              model_configs: List[Dict],
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              max_workers: int = 2) -> Dict:
        """
        ADVANCED FEATURE: Train multiple AI models simultaneously
        
        BUSINESS BENEFIT:
        - Instead of training models one by one (sequential), train them at the same time (parallel)
        - Reduces total training time significantly
        - Like having multiple workers build different parts of a house simultaneously
        
        PROCESS:
        1. Creates separate training jobs for each model
        2. Runs multiple training jobs at the same time
        3. Collects results as each job completes
        4. Returns all trained models and their performance stats
        
        Parameters:
        - model_configs: List of model configurations to train
        - X_train: Training data features
        - y_train: Training data targets (what we want to predict)
        - max_workers: Maximum number of models to train simultaneously
        
        Returns:
        - Dictionary containing all trained models and training results
        """
        
        # Initialize result containers
        trained_models = {}      # Successfully trained models
        training_results = {}    # Training statistics and results
        
        def train_single_model(config: Dict) -> tuple:
            """
            INNER FUNCTION: Trains one individual model
            
            This function runs in parallel - multiple copies run simultaneously
            """
            model_name = config['name']
            model_class = config['class']
            params = config.get('params', {})
            
            try:
                # Initialize the model with specified parameters
                model = model_class(**params)
                
                # Record training start time
                start_time = datetime.now()
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Calculate training time
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Return success result
                return (model_name, model, {
                    'training_time': training_time, 
                    'status': 'success'
                })
                
            except Exception as e:
                # Return failure result
                return (model_name, None, {
                    'error': str(e), 
                    'status': 'failed'
                })
        
        # PARALLEL EXECUTION: Use ThreadPoolExecutor to run training jobs simultaneously
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            # STEP 1: Submit all training jobs to the executor
            future_to_config = {
                executor.submit(train_single_model, config): config 
                for config in model_configs
            }
            
            # STEP 2: Collect results as each job completes
            for future in as_completed(future_to_config):
                model_name, model, result = future.result()
                
                if model is not None:
                    # Training succeeded
                    trained_models[model_name] = model
                    training_results[model_name] = result
                    
                    # Update progress if callback is available
                    if self.progress_callback:
                        completed = len(trained_models)
                        total = len(model_configs)
                        self.progress_callback(completed, total, f"Trained {model_name}")
                else:
                    # Training failed
                    training_results[model_name] = result
                    print(f"Failed to train {model_name}: {result.get('error', 'Unknown error')}")
        
        # Return comprehensive results
        return {
            'models': trained_models,
            'results': training_results,
            'summary': {
                'total_models': len(model_configs),
                'successful': len(trained_models),
                'failed': len(model_configs) - len(trained_models)
            }
        }
    
    # --------------------------------------------------------------------------
    # PREDICTION PIPELINE CREATOR
    # --------------------------------------------------------------------------
    def create_prediction_pipeline(self, 
                                 models: Dict,
                                 preprocessing_steps: List[Callable],
                                 postprocessing_steps: List[Callable] = None) -> Callable:
        """
        PIPELINE BUILDER: Creates a complete end-to-end prediction system
        
        WHAT IT CREATES:
        A single function that can take raw data and output final predictions
        
        PIPELINE STEPS:
        1. Data Preprocessing (cleaning, formatting)
        2. Model Predictions (applying trained models)
        3. Postprocessing (formatting results, calculations)
        
        BUSINESS ANALOGY:
        Like setting up an assembly line in a factory where raw materials 
        go through multiple stations and come out as finished products
        
        Parameters:
        - models: Dictionary of trained models to use
        - preprocessing_steps: List of data preparation functions
        - postprocessing_steps: List of result formatting functions
        
        Returns:
        - A single function that executes the complete pipeline
        """
        
        def pipeline(data: pd.DataFrame) -> pd.DataFrame:
            """
            THE ACTUAL PIPELINE: Executes all steps in sequence
            
            This is the function that gets returned - it contains all the 
            logic to process data from start to finish
            """
            processed_data = data.copy()
            
            # STAGE 1: Apply preprocessing steps
            for step_num, step in enumerate(preprocessing_steps):
                try:
                    processed_data = step(processed_data)
                except Exception as e:
                    print(f"Preprocessing step {step_num + 1} failed: {e}")
                    continue  # Skip failed steps and continue with remaining ones
            
            # STAGE 2: Make predictions with all models
            for model_name, model in models.items():
                try:
                    # Select numeric columns for prediction
                    numeric_cols = processed_data.select_dtypes(include=[np.number])
                    predictions = model.predict(numeric_cols)
                    processed_data[f'{model_name}_prediction'] = predictions
                except Exception as e:
                    print(f"Prediction failed for {model_name}: {e}")
                    processed_data[f'{model_name}_prediction'] = np.nan
            
            # STAGE 3: Apply postprocessing steps
            if postprocessing_steps:
                for step_num, step in enumerate(postprocessing_steps):
                    try:
                        processed_data = step(processed_data)
                    except Exception as e:
                        print(f"Postprocessing step {step_num + 1} failed: {e}")
                        continue
            
            return processed_data
        
        return pipeline
    
    # --------------------------------------------------------------------------
    # MULTI-FORMAT EXPORT
    # --------------------------------------------------------------------------
    def export_results_multiple_formats(self, 
                                      data: pd.DataFrame,
                                      base_filename: str,
                                      formats: List[str] = None) -> Dict[str, str]:
        """
        EXPORT UTILITY: Saves results in multiple file formats simultaneously
        
        BUSINESS PURPOSE:
        - Different stakeholders prefer different formats
        - Excel for business users, CSV for technical users, JSON for web applications
        - Saves time by creating all formats at once
        
        Parameters:
        - data: The data to export
        - base_filename: Base name for files (without extension)
        - formats: List of desired formats
        
        Returns:
        - Dictionary mapping each format to its saved file path
        """
        if formats is None:
            formats = ['csv', 'excel']  # Default formats
        
        exported_files = {}
        
        # Export in each requested format
        for fmt in formats:
            try:
                if fmt == 'csv':
                    filepath = f"{base_filename}.csv"
                    data.to_csv(filepath, index=False)
                    exported_files['csv'] = filepath
                    
                elif fmt == 'excel':
                    filepath = f"{base_filename}.xlsx"
                    data.to_excel(filepath, index=False)
                    exported_files['excel'] = filepath
                    
                elif fmt == 'json':
                    filepath = f"{base_filename}.json"
                    data.to_json(filepath, orient='records', indent=2)
                    exported_files['json'] = filepath
                    
                elif fmt == 'pickle':
                    filepath = f"{base_filename}.pickle"
                    data.to_pickle(filepath)
                    exported_files['pickle'] = filepath
                    
            except Exception as e:
                print(f"Failed to export as {fmt}: {e}")
        
        return exported_files
    
    # --------------------------------------------------------------------------
    # PROCESSING REPORT GENERATOR
    # --------------------------------------------------------------------------
    def create_processing_report(self, 
                               processing_stats: Dict,
                               save_path: str = None) -> str:
        """
        REPORTING TOOL: Creates detailed performance reports
        
        BUSINESS VALUE:
        - Shows processing efficiency metrics
        - Helps identify bottlenecks
        - Provides documentation for audit purposes
        - Demonstrates system performance to stakeholders
        
        Parameters:
        - processing_stats: Statistics from processing operations
        - save_path: Optional path to save the report
        
        Returns:
        - Formatted report as a string
        """
        
        # REPORT HEADER
        report = f"""
BATCH PROCESSING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

PROCESSING SUMMARY:
"""
        
        # CALCULATE OVERALL STATISTICS
        total_rows = sum(stats.get('total_rows', 0) for stats in processing_stats.values())
        total_time = sum(stats.get('processing_time', 0) for stats in processing_stats.values())
        
        # SUMMARY SECTION
        report += f"  Total files processed: {len(processing_stats)}\n"
        report += f"  Total rows processed: {total_rows:,}\n"
        report += f"  Total processing time: {total_time:.2f} seconds\n"
        if total_time > 0:
            report += f"  Average throughput: {total_rows/total_time:.0f} rows/second\n\n"
        
        # INDIVIDUAL FILE DETAILS
        report += "INDIVIDUAL FILE STATISTICS:\n"
        for file_path, stats in processing_stats.items():
            filename = os.path.basename(file_path)
            report += f"\n📊 {filename}:\n"
            report += f"   Rows processed: {stats.get('total_rows', 0):,}\n"
            report += f"   Processing time: {stats.get('processing_time', 0):.2f}s\n"
            report += f"   Throughput: {stats.get('rows_per_second', 0):.0f} rows/sec\n"
            report += f"   Chunks processed: {stats.get('chunks_processed', 0)}\n"
            
            if 'output_path' in stats:
                report += f"   Output saved to: {stats['output_path']}\n"
        
        report += "\n" + "="*60
        
        # SAVE REPORT IF REQUESTED
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write(report)
                print(f"Processing report saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save report: {e}")
        
        return report


# ==============================================================================
# PROGRESS TRACKER CLASS
# ==============================================================================
class ProgressTracker:
    """
    THREAD-SAFE PROGRESS TRACKING: Handles progress updates in multi-threaded operations
    
    WHY WE NEED THIS:
    - When multiple processes run simultaneously, they need to safely share progress info
    - Prevents conflicts when multiple threads try to update progress at the same time
    - Provides a centralized way to track and report progress
    
    REAL-WORLD ANALOGY:
    Like having a single scoreboard that multiple teams can safely update 
    their scores on without interfering with each other
    """
    
    def __init__(self):
        """Initialize progress tracker with thread-safe defaults"""
        self._lock = threading.Lock()    # Ensures thread safety
        self._progress = 0               # Current progress
        self._total = 100               # Total items to process
        self._message = ""              # Current status message
        self._callbacks = []            # Functions to call on progress updates
    
    def set_total(self, total: int):
        """
        SET EXPECTATIONS: Define how many total items will be processed
        
        Parameters:
        - total: Total number of items to process
        """
        with self._lock:  # Thread-safe update
            self._total = total
    
    def update(self, increment: int = 1, message: str = ""):
        """
        UPDATE PROGRESS: Increment progress and notify listeners
        
        Parameters:
        - increment: How much to increase progress by
        - message: Status message to display
        """
        with self._lock:  # Thread-safe update
            self._progress = min(self._progress + increment, self._total)
            if message:
                self._message = message
            
            # Notify all registered callback functions
            for callback in self._callbacks:
                try:
                    callback(self._progress, self._total, self._message)
                except Exception as e:
                    print(f"Progress callback error: {e}")
    
    def add_callback(self, callback: Callable[[int, int, str], None]):
        """
        REGISTER LISTENER: Add a function to be called on progress updates
        
        Parameters:
        - callback: Function that will receive progress updates
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def get_progress(self) -> tuple:
        """
        GET CURRENT STATUS: Return current progress information
        
        Returns:
        - Tuple of (current_progress, total, message)
        """
        with self._lock:
            return self._progress, self._total, self._message
    
    def reset(self):
        """RESET PROGRESS: Start over with clean progress tracking"""
        with self._lock:
            self._progress = 0
            self._message = ""


# ==============================================================================
# DEMONSTRATION AND TESTING CODE
# ==============================================================================
if __name__ == "__main__":
    """
    EXAMPLE USAGE: Demonstrates how to use the batch processor
    
    This section runs when the file is executed directly (not imported)
    It shows practical examples of how to use all the features
    """
    
    print("="*60)
    print("BATCH PROCESSOR DEMONSTRATION")
    print("="*60)
    
    # STEP 1: Create a batch processor instance
    processor = BatchProcessor(chunk_size=1000)
    
    # STEP 2: Define a sample processing function
    def sample_processing_func(chunk):
        """
        EXAMPLE PROCESSING: Adds a 'processed' column to show the data was handled
        
        In real applications, this would contain your business logic:
        - Data cleaning
        - Calculations
        - Feature engineering
        - Quality checks
        """
        chunk['processed'] = True
        chunk['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return chunk
    
    # STEP 3: Define progress callback for user feedback
    def progress_callback(current, total, message):
        """PROGRESS DISPLAY: Shows progress updates to the user"""
        percentage = (current / total * 100) if total > 0 else 0
        print(f"Progress: {current}/{total} ({percentage:.1f}%) - {message}")
    
    # STEP 4: Set up progress tracking
    processor.set_progress_callback(progress_callback)
    
    # STEP 5: Create sample data for demonstration
    print("Creating sample dataset...")
    sample_data = pd.DataFrame({
        'customer_id': range(1, 10001),
        'feature1': np.random.randn(10000),
        'feature2': np.random.randn(10000),
        'target': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
    # Save sample data
    sample_file = 'sample_large_dataset.csv'
    sample_data.to_csv(sample_file, index=False)
    print(f"Sample dataset created with {len(sample_data):,} rows")
    
    # STEP 6: Process the large dataset
    print("\nStarting batch processing...")
    result = processor.process_large_dataset(
        file_path=sample_file,
        processing_func=sample_processing_func,
        output_path='processed_dataset.csv'
    )
    
    # STEP 7: Display results
    if 'error' not in result:
        print("\n" + "="*50)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        stats = result.get('stats', {})
        print(f"📊 Rows processed: {stats.get('total_rows', 0):,}")
        print(f"⏱️  Processing time: {stats.get('processing_time', 0):.2f} seconds")
        print(f"🚀 Throughput: {stats.get('rows_per_second', 0):.0f} rows/second")
        print(f"🔧 Chunks processed: {stats.get('chunks_processed', 0)}")
        
        # Create and display processing report
        report = processor.create_processing_report(processor.processing_stats)
        print("\n" + report)
        
    else:
        print(f"❌ Processing failed: {result['error']}")
    
    # STEP 8: Clean up demonstration files
    cleanup_files = [sample_file, 'processed_dataset.csv']
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🧹 Cleaned up: {file}")
    
    print("\n✅ Demonstration completed!")