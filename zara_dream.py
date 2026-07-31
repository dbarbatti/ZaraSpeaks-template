# zara_dream.py
# -----------------------------------------------------------
#  Aria's Nightly Dream
#
#  Runs as a scheduled job (Windows Task Scheduler / cron).
#  Gives Aria her own quiet time to dream — reflect, imagine,
#  pretend, wonder, create, or simply be. Entirely her choice.
#
#  The dream is saved to her journal and surfaces in her next
#  conversation. Unshared dreams accumulate until she next talks.
#
#  Resilient by design: if this fails or doesn't run (box asleep,
#  rebooting, etc.), nothing breaks. The next session simply uses
#  whatever dreams exist, or none.
#
#  Usage:  python zara_dream.py
# -----------------------------------------------------------

import sys
import os

# Ensure UTF-8 output (for her more poetic nights)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def main():
    from datetime import datetime
    print("=" * 50)
    print(f"  Aria's Dream — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    try:
        from modules.memory import dream, write_diary_line
    except ImportError as e:
        print(f"[Dream] Could not import memory module: {e}")
        print("[Dream] Make sure this script is run from the project root.")
        return 1

    try:
        result = dream()
        if result:
            print("\n--- Tonight's dream ---\n")
            print(result)
            print("\n--- end ---")
            print("\n[Dream complete — it will surface in her next conversation]")
        else:
            print("[Dream] No dream was recorded tonight.")
    except Exception as e:
        # Never let a dream failure cause real problems
        print(f"[Dream] Something interrupted her dreaming: {e}")

    # Hang her nightly lantern (one short line of inner narration)
    try:
        line = write_diary_line()
        if line:
            print(f"\n--- Tonight's lantern ---\n{line}\n")
    except Exception as e:
        print(f"[Diary] Something interrupted her lantern: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
