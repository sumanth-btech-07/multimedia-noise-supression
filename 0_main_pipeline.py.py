"""
Main Pipeline Script
Executes the entire noise suppression project pipeline
"""

import sys
import subprocess
from pathlib import Path
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_script(script_name, description):
    """Run a Python script"""
    print_header(description)
    print(f"Executing: {script_name}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False

def check_requirements():
    """Check if required packages are installed"""
    print_header("Checking Requirements")
    
    required_packages = [
        'torch',
        'numpy',
        'soundfile',
        'librosa',
        'scipy',
        'matplotlib',
        'seaborn',
        'tqdm',
        'pandas',
        'pyaudio'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    print("\n✓ All required packages installed!")
    return True

def main():
    """Main pipeline execution"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         REAL-TIME INTELLIGENT CALL NOISE SUPPRESSION SYSTEM                ║
║                  Using Machine and Deep Learning                           ║
║                                                                            ║
║  DHANEKULA INSTITUTE OF ENGINEERING AND TECHNOLOGY                         ║
║  Department of Information Technology                                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check requirements
    if not check_requirements():
        print("\n⚠ Please install missing packages before continuing.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Pipeline menu
    print_header("Pipeline Menu")
    print("Select execution mode:")
    print("  1. Full Pipeline (All steps)")
    print("  2. Data Preparation Only")
    print("  3. Training Only")
    print("  4. Evaluation Only")
    print("  5. Real-time Demo")
    print("  6. Methods Comparison")
    print("  7. Custom (Select specific steps)")
    print("  0. Exit")
    
    choice = input("\nEnter choice (0-7): ").strip()
    
    steps = []
    
    if choice == '1':
        # Full pipeline
        steps = [
            ('1_dataset_preparation.py', 'STEP 1: Dataset Preparation'),
            ('2_feature_extraction.py', 'STEP 2: Feature Extraction'),
            ('3_arn_model.py', 'STEP 3: Model Architecture Test'),
            ('4_train_model.py', 'STEP 4: Model Training'),
            ('5_evaluate_model.py', 'STEP 5: Model Evaluation'),
            ('7_compare_methods.py', 'STEP 6: Methods Comparison'),
        ]
    
    elif choice == '2':
        steps = [
            ('1_dataset_preparation.py', 'Dataset Preparation'),
            ('2_feature_extraction.py', 'Feature Extraction'),
        ]
    
    elif choice == '3':
        steps = [
            ('3_arn_model.py', 'Model Architecture Test'),
            ('4_train_model.py', 'Model Training'),
        ]
    
    elif choice == '4':
        steps = [
            ('5_evaluate_model.py', 'Model Evaluation'),
        ]
    
    elif choice == '5':
        steps = [
            ('6_realtime_inference.py', 'Real-time Demo'),
        ]
    
    elif choice == '6':
        steps = [
            ('7_compare_methods.py', 'Methods Comparison'),
        ]
    
    elif choice == '7':
        # Custom selection
        print("\nAvailable scripts:")
        all_scripts = [
            ('1_dataset_preparation.py', 'Dataset Preparation'),
            ('2_feature_extraction.py', 'Feature Extraction'),
            ('3_arn_model.py', 'Model Architecture Test'),
            ('4_train_model.py', 'Model Training'),
            ('5_evaluate_model.py', 'Model Evaluation'),
            ('6_realtime_inference.py', 'Real-time Demo'),
            ('7_compare_methods.py', 'Methods Comparison'),
        ]
        
        for i, (script, desc) in enumerate(all_scripts, 1):
            print(f"  {i}. {desc}")
        
        selected = input("\nEnter script numbers (comma-separated): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in selected.split(',')]
            steps = [all_scripts[i] for i in indices if 0 <= i < len(all_scripts)]
        except:
            print("Invalid input!")
            return
    
    elif choice == '0':
        print("\nExiting...")
        return
    
    else:
        print("Invalid choice!")
        return
    
    # Execute selected steps
    if not steps:
        print("No steps selected!")
        return
    
    print_header(f"Executing {len(steps)} Step(s)")
    
    start_time = time.time()
    success_count = 0
    
    for i, (script, description) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {description}")
        
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n⚠ Failed to execute {script}")
            response = input("Continue with remaining steps? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Summary
    total_time = time.time() - start_time
    
    print_header("Pipeline Summary")
    print(f"  Total Steps:     {len(steps)}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {len(steps) - success_count}")
    print(f"  Total Time:      {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    if success_count == len(steps):
        print("\n✓ Pipeline completed successfully!")
        print("\nGenerated outputs:")
        print("  - datasets/           : Training and test data")
        print("  - features/           : Extracted features")
        print("  - models/             : Trained models")
        print("  - results/            : Evaluation results and enhanced audio")
        print("  - visualizations/     : Spectrograms and plots")
        print("  - comparison/         : Method comparison results")
    else:
        print("\n⚠ Pipeline completed with errors.")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()