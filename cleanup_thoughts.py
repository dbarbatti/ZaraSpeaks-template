# cleanup_thoughts.py
# -----------------------------------------------------------
#  Delete memory files that contain "thought for [day], [date]"
#  These are auto-generated "thought of the day" sessions
# -----------------------------------------------------------

import os
import re
import json
from config import MEMORY_DIR

def cleanup():
    pattern = re.compile(r"thought for \w+day", re.IGNORECASE)
    
    files = [f for f in os.listdir(MEMORY_DIR) 
             if f.startswith("zara_memory_") and f.endswith(".json")]
    
    print(f"Scanning {len(files)} memory files...")
    
    to_delete = []
    to_keep = []
    
    for fname in sorted(files):
        path = os.path.join(MEMORY_DIR, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if pattern.search(content):
                to_delete.append(fname)
            else:
                to_keep.append(fname)
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
            to_keep.append(fname)
    
    print(f"\nFound {len(to_delete)} 'thought of the day' files to delete")
    print(f"Keeping {len(to_keep)} real conversation files")
    
    if not to_delete:
        print("Nothing to delete!")
        return
    
    print(f"\nFiles to DELETE:")
    for f in to_delete[:10]:
        print(f"  {f}")
    if len(to_delete) > 10:
        print(f"  ... and {len(to_delete) - 10} more")
    
    confirm = input(f"\nDelete {len(to_delete)} files? (y/n): ").strip().lower()
    if confirm == 'y':
        deleted = 0
        for fname in to_delete:
            try:
                os.remove(os.path.join(MEMORY_DIR, fname))
                deleted += 1
            except Exception as e:
                print(f"  Error deleting {fname}: {e}")
        print(f"\nDeleted {deleted} files. {len(to_keep)} files remaining.")
    else:
        print("Cancelled.")

if __name__ == "__main__":
    cleanup()
