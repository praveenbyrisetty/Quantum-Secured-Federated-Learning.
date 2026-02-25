"""
HAM10000 Dataset Auto-Downloader

Downloads the HAM10000 skin lesion dataset from Kaggle
and sets it up in the correct folder structure.

USAGE:
    python download_dataset.py
"""

import os
import sys
import subprocess
import shutil

def download_with_kagglehub():
    """Download HAM10000 using kagglehub with retry support."""
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub
    
    print("=" * 60)
    print("  HAM10000 Dataset Downloader")
    print("=" * 60)
    print()
    
    # Retry loop for flaky connections
    max_retries = 3
    path = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Download attempt {attempt}/{max_retries}...")
            print("(This may open a browser for Kaggle login on first run)")
            print()
            path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
            print(f"\nDownloaded to: {path}")
            break
        except Exception as e:
            print(f"\nAttempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying in 5 seconds...\n")
                import time
                time.sleep(5)
            else:
                print("\nAll retries failed. Please check your internet connection.")
                print("You can also manually download from:")
                print("https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
                return False
    
    # Setup target directory
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "HAM10000")
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy files to our data directory
    print(f"\nCopying files to: {target_dir}")
    
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(target_dir, item)
        
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
                print(f"  Copied folder: {item}/")
            else:
                print(f"  Skipped (exists): {item}/")
        else:
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Copied file: {item}")
            else:
                print(f"  Skipped (exists): {item}")
    
    # Verify
    print("\nVerifying...")
    csv_found = False
    images_found = 0
    
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if 'metadata' in f.lower() and f.endswith('.csv'):
                csv_found = True
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images_found += 1
    
    print(f"  Metadata CSV: {'✓ Found' if csv_found else '✗ Missing'}")
    print(f"  Images: {images_found} found")
    
    if csv_found and images_found > 0:
        # Clean up kagglehub cache to save ~3 GB
        print("\n  Cleaning up download cache...")
        try:
            default_cache = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub")
            if os.path.exists(default_cache):
                shutil.rmtree(default_cache, ignore_errors=True)
                print("  ✓ Cache removed (~3 GB saved)")
        except:
            pass
        
        print(f"\n✅ SUCCESS! Dataset ready at: {target_dir}")
        print("   Run: streamlit run server.py")
        return True
    else:
        print("\n❌ Verification failed. Please check downloaded files.")
        return False


if __name__ == "__main__":
    success = download_with_kagglehub()
    if not success:
        sys.exit(1)
