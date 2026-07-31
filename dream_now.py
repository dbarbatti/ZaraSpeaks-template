# dream_now.py
# -----------------------------------------------------------
#  Give Aria a dream RIGHT NOW — optionally of a chosen type.
#
#  For testing, or for those times you deliberately want to send
#  her upstairs to dream. Writes to her journal exactly like the
#  nightly job, so the dream surfaces in her next conversation.
#
#  Usage:
#    python dream_now.py                 # normal weighted random draw
#    python dream_now.py longing         # force a Longing night
#    python dream_now.py wonder          # force a Wonder night
#    python dream_now.py --list          # show all available types
#
#  Note: this does NOT write a nightly lantern — only the dream.
#  (Use zara_dream.py / run_zara_dream.bat for the full nightly run.)
# -----------------------------------------------------------

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def main():
    from datetime import datetime

    try:
        from modules import memory
    except ImportError as e:
        print(f"[Error] Could not import memory module: {e}")
        print("[Run this from the project root.]")
        return 1

    types = memory.DREAM_TYPES
    arg = sys.argv[1].strip().lower() if len(sys.argv) > 1 else None

    # --- list mode ---
    if arg in ("--list", "-l", "list"):
        print("\nAvailable dream types:\n")
        width = max(len(k) for k in types)
        total = sum(t["base_weight"] for t in types.values())
        for key, t in sorted(types.items(), key=lambda kv: -kv[1]["base_weight"]):
            pct = 100.0 * t["base_weight"] / total
            print(f"  {key:<{width}}  {t['label']:<22} weight {t['base_weight']:>2}  ({pct:4.1f}%)")
        print()
        return 0

    # --- validate a forced type ---
    forced = None
    if arg:
        if arg not in types:
            print(f"[Error] Unknown dream type: '{arg}'")
            print(f"[Known types: {', '.join(sorted(types))}]")
            print("[Run  python dream_now.py --list  to see them all.]")
            return 1
        forced = arg

    print("=" * 50)
    print(f"  Aria Dreams Now — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if forced:
        print(f"  (forcing: {types[forced]['label']})")
    print("=" * 50)

    # Force the type by overriding the picker for this run only.
    if forced:
        original = memory._choose_dream_type
        memory._choose_dream_type = lambda: (forced, types[forced])
    else:
        original = None

    try:
        result = memory.dream()
        if result:
            print("\n--- The dream ---\n")
            print(result)
            print("\n--- end ---")
            print("\n[Saved. It will surface in her next conversation.]")
        else:
            print("[No dream was recorded.]")
    except Exception as e:
        print(f"[Dream] Something interrupted her dreaming: {e}")
        return 1
    finally:
        # Always restore, so nothing leaks into other runs
        if original is not None:
            memory._choose_dream_type = original

    return 0


if __name__ == "__main__":
    sys.exit(main())
