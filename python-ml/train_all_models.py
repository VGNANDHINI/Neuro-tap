"""
Master Training Script
Trains all models sequentially
"""

import os
import subprocess
import sys

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# List of training scripts
training_scripts = [
    'spiral_svm_model.py',
    'tapping_rf_model.py',
    'reaction_lr_model.py',
    'voice_svm_model.py',
    'nonmotor_nb_model.py',
    'cognitive_gb_model.py',
    'ensemble_final_model.py'
]

print("=" * 60)
print("PARKINSON'S DISEASE PREDICTION - MODEL TRAINING")
print("=" * 60)

for i, script in enumerate(training_scripts, 1):
    print(f"\n[{i}/{len(training_scripts)}] Training: {script}")
    print("-" * 60)
    
    try:
        result = subprocess.run([sys.executable, script], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"✓ {script} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error training {script}")
        print(f"Error: {e}")
        continue
    except FileNotFoundError:
        print(f"✗ Script not found: {script}")
        continue

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("\nAll models have been trained and saved to the 'models/' directory")
print("You can now start the Flask API server using: python app.py")