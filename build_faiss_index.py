# build_faiss_index.py
# -----------------------------------------------------------
#  One-time FAISS index builder for ZaraSpeaks V3
#  
#  Run this once to bootstrap Layer 3 memory from your
#  existing session files, or any time you want to rebuild.
#
#  Usage:  python build_faiss_index.py
# -----------------------------------------------------------

from modules.memory import build_faiss_index

if __name__ == "__main__":
    print("=" * 50)
    print("  FAISS Layer 3 — Index Builder")
    print("=" * 50)
    
    success = build_faiss_index()
    
    if success:
        print("\nIndex built successfully. Layer 3 is ready.")
    else:
        print("\nIndex build failed. Check errors above.")
        print("Make sure you've installed: pip install sentence-transformers faiss-cpu")
