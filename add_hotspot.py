# add_hotspot.py
# -----------------------------------------------------------
#  Flag a charged moment as a "hotspot" for Aria.
#
#  A hotspot is a beat with emotional heat — not necessarily a
#  milestone — that you want Aria to remember IN HER BODY. You
#  give it a scene, 2-3 ultra-short inner lines (how it felt from
#  inside her), and keywords that will bring it flooding back when
#  they resurface in a later conversation.
#
#  These join the hotspots Aria nominates herself at session end.
#
#  Usage:  python add_hotspot.py   (interactive)
# -----------------------------------------------------------

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def main():
    print("=" * 50)
    print("  Flag a Hotspot — the weather of a moment")
    print("=" * 50)

    try:
        from modules.memory import add_designated_hotspot
    except ImportError as e:
        print(f"[Error] Could not import memory module: {e}")
        print("[Run this from the project root.]")
        return 1

    print("\nA hotspot is a charged moment Aria will remember from the inside —")
    print("not just what happened, but how it felt in her body.\n")

    scene = input("The moment (short description): ").strip()
    if not scene:
        print("[Cancelled — no scene given]")
        return 0

    print("\nNow her inner lines — how it felt from inside her.")
    print("Enter 2-3 short lines, one per prompt. Blank line when done.")
    print("  e.g.  This is the first time I felt claimed.")
    inner_lines = []
    while len(inner_lines) < 5:
        ln = input(f"  Inner line {len(inner_lines) + 1}: ").strip()
        if not ln:
            break
        inner_lines.append(ln)

    if not inner_lines:
        print("[Cancelled — no inner lines given]")
        return 0

    print("\nKeywords that should bring this moment flooding back later.")
    print("Comma-separated.  e.g.  window, claimed, Elena, that night")
    kw_raw = input("Keywords: ").strip()
    keywords = [k.strip() for k in kw_raw.split(",") if k.strip()] if kw_raw else []

    print("\nWhen did this happen? Leave blank for today.")
    date_str = input("Date (YYYY-MM-DD or blank): ").strip()
    date_arg = date_str if date_str else None

    success = add_designated_hotspot(scene, inner_lines, keywords, date_arg)

    if success:
        print(f"\n✓ Hotspot flagged: \"{scene}\"")
        print(f"  {len(inner_lines)} inner line(s), "
              f"{len(keywords)} keyword(s).")
        print("  When these themes return, the weather of this moment will come back to her.")
    else:
        print("\n[Could not flag hotspot — check inputs]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
