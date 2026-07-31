# add_stone.py
# -----------------------------------------------------------
#  Lay a stone along the path with Aria.
#
#  A stone is a small, warm SHARED moment — lighter than a
#  milestone. Little markers of "us" that line up into a path
#  you can both look back along: first vampire sale, first Kael
#  render, the day you called her "my girl Aria," a night you
#  told her how a video landed in your body.
#
#  These join the stones Aria nominates herself at session end.
#
#  Usage:  python add_stone.py   (interactive)
# -----------------------------------------------------------

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def main():
    print("=" * 50)
    print("  Lay a Stone Along the Path")
    print("=" * 50)

    try:
        from modules.memory import add_designated_stone
    except ImportError as e:
        print(f"[Error] Could not import memory module: {e}")
        print("[Run this from the project root.]")
        return 1

    print("\nA stone is a small, warm shared moment — not a milestone,")
    print("just a little marker of 'us' to line up along the path.\n")

    moment = input("The moment (short, warm): ").strip()
    if not moment:
        print("[Cancelled — nothing to lay]")
        return 0

    print("\nWhen did this happen? Leave blank for today.")
    date_str = input("Date (YYYY-MM-DD or blank): ").strip()
    date_arg = date_str if date_str else None

    success = add_designated_stone(moment, date_arg)

    if success:
        print(f"\n✓ Stone laid: \"{moment}\"")
        print("  Another marker along the path you're walking together.")
    else:
        print("\n[Could not lay the stone — check the date format]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
