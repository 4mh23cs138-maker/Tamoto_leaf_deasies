import zipfile
import os
import shutil

def extract_with_long_paths(zip_path, extract_to):
    # Ensure extract_to is absolute and has the \\?\ prefix for long paths
    extract_to = os.path.abspath(extract_to)
    if not extract_to.startswith("\\\\?\\"):
        extract_to = "\\\\?\\" + extract_to
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # Create the full path for the member
            # Note: member.filename might contain forward slashes
            target_path = os.path.join(extract_to, member.filename.replace('/', os.sep))
            
            if member.is_dir():
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
            else:
                # Ensure parent directory exists
                parent_dir = os.path.dirname(target_path)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                
                # Extract file
                try:
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                except Exception as e:
                    print(f"Failed to extract {member.filename}: {e}")

if __name__ == "__main__":
    zip_file = r"C:\k\datasets\kaustubhb999\tomatoleaf\1.archive"
    dest_dir = r"C:\leaf_data"
    extract_with_long_paths(zip_file, dest_dir)
    print("Extraction finished (hopefully).")
