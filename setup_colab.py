import os
import sys
import shutil

def setup_colab_environment():
    """Set up the Colab environment for training."""
    # Create necessary directories
    os.makedirs('/content/XrayDetector', exist_ok=True)
    
    # Copy files from Drive to Colab
    source_dir = '/content/drive/MyDrive/XrayDetector'
    target_dir = '/content/XrayDetector'
    
    # Copy all files and directories
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)
        
        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)
    
    # Add the project directory to Python path
    sys.path.append(target_dir)
    
    # Verify the directory structure
    print("\nVerifying directory structure:")
    print(f"Contents of {target_dir}:")
    for root, dirs, files in os.walk(target_dir):
        level = root.replace(target_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
    
    print("\nColab environment setup complete!")
    print(f"Project files copied to: {target_dir}")
    print("Python path updated to include project root")

if __name__ == '__main__':
    setup_colab_environment() 