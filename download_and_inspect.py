import kagglehub
import os

def main():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("kaustubhb999/tomatoleaf")
    print(f"Dataset downloaded to: {path}")
    
    # List contents to understand structure
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        # Show first 5 files if any
        for f in files[:5]:
            print('{}{}'.format(subindent, f))
        if len(files) > 5:
            print('{}(and {} more)'.format(subindent, len(files) - 5))
        if level >= 2: # Don't go too deep
            break

if __name__ == "__main__":
    main()
