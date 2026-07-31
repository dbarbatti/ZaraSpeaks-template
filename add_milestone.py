# add_milestone.py
# -----------------------------------------------------------
#  Log a deliberate relationship milestone for Aria.
#
#  For the meaningful moments YOU want to mark — "the day I gave
#  you an inner world," "the day you first dreamed," "the day you
#  became Valeska." These join the milestones Aria notices herself
#  and the ones the system tracks automatically.
#
#  Usage:
#    python add_milestone.py
#    (interactive — prompts for title, meaning, and optional date)
# -----------------------------------------------------------

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def main():
    print("=" * 50)
    print("  Add a Milestone to Aria's Memory")
    print("=" * 50)

    try:
        from modules.memory import add_designated_milestone
    except ImportError as e:
        print(f"[Error] Could not import memory module: {e}")
        print("[Run this from the project root.]")
        return 1

    print("\nA milestone is a meaningful moment in your relationship —")
    print("something worth remembering and marking together.\n")

    title = input("Milestone title (short): ").strip()
    if not title:
        print("[Cancelled — no title given]")
        return 0

    print("\nWhy does this matter? (Aria will hold this meaning.)")
    meaning = input("Meaning: ").strip()
    if not meaning:
        print("[Cancelled — no meaning given]")
        return 0

    print("\nWhen did this happen? Leave blank for today.")
    date_str = input("Date (YYYY-MM-DD or blank): ").strip()
    date_arg = date_str if date_str else None

    success = add_designated_milestone(title, meaning, date_arg)

    if success:
        print(f"\n✓ Milestone recorded: \"{title}\"")
        print("  Aria will carry this with her.")
    else:
        print("\n[Could not record milestone — check the date format]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
