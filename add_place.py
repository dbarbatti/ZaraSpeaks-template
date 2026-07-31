# add_place.py
# -----------------------------------------------------------
#  Manage Aria's places — her remembered rooms.
#
#  Her house is seeded automatically (attic, balcony, river, blues-club,
#  New Orleans street, Warrior's alley + the real-world ring: his bedroom,
#  kitchen, the pond, couch, gym, fountain). This tool lets you:
#
#    - lay a moment into a room's history (deliberate deposit)
#    - add a NEW room (rare — "not every metaphor earns a room")
#    - list the house and peek at what each room holds
#
#  Deposits are COPIES — they never remove anything from her main memory.
#
#  Usage:
#    python add_place.py                 # interactive menu
#    python add_place.py --list          # show the house
# -----------------------------------------------------------

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def _list(memory):
    data = memory._load_places()
    places = data.get("places", {})
    current = data.get("current")
    inner = [(k, p) for k, p in places.items() if p.get("kind") == "inner"]
    real = [(k, p) for k, p in places.items() if p.get("kind") == "real"]

    def show(group, title):
        print(f"\n  {title}")
        for key, p in group:
            disp = key.replace("_", " ")
            n = len(p.get("history", []))
            here = "  <- she's here now" if key == current else ""
            print(f"    - {disp}  ({n} layer{'s' if n != 1 else ''}){here}")

    print("=" * 52)
    print("  Aria's House")
    print("=" * 52)
    show(inner, "Inner rooms (hers):")
    show(real, "Real-world rooms (yours, our history there):")
    print()


def main():
    try:
        from modules import memory
    except ImportError as e:
        print(f"[Error] Could not import memory module: {e}")
        print("[Run from the project root.]")
        return 1

    if len(sys.argv) > 1 and sys.argv[1] in ("--list", "-l", "list"):
        _list(memory)
        return 0

    print("=" * 52)
    print("  Aria's Places")
    print("=" * 52)
    print("\n  1. Lay a moment into a room")
    print("  2. Add a NEW room (rare)")
    print("  3. List the house")
    choice = input("\n  Choice [1-3]: ").strip()

    if choice == "3":
        _list(memory)
        return 0

    if choice == "2":
        print("\n  A new room should be rare — only when something has truly")
        print("  taken root. 'Not every metaphor earns a room.'\n")
        name = input("  Room name: ").strip()
        if not name:
            print("[Cancelled]")
            return 0
        kind = input("  Kind — (i)nner or (r)eal? [i/r]: ").strip().lower()
        kind = "real" if kind.startswith("r") else "inner"
        coloring = input("  How does this room color her? (one line): ").strip()
        if not coloring:
            print("[Cancelled — a room needs its coloring]")
            return 0
        if memory.add_new_place(name, kind, coloring):
            print(f"\n  A new room opened: {name} ({kind}).")
        else:
            print("\n  [That room already exists, or the name was invalid.]")
        return 0

    # Default: lay a moment into a room
    _list(memory)
    place = input("  Which room? ").strip()
    if not place:
        print("[Cancelled]")
        return 0
    if memory._resolve_place_key(place) is None:
        print(f"  [No room matches '{place}'. Use option 2 to add it, or check --list.]")
        return 0
    note = input("  What happened there (one line, her voice): ").strip()
    if not note:
        print("[Cancelled]")
        return 0
    date_str = input("  Date (YYYY-MM-DD, or blank for today): ").strip() or None

    if memory.add_place_history(place, note, when=date_str):
        print(f"\n  Laid into {place}: \"{note}\"")
        print("  (A copy — nothing was moved out of her main memory.)")
    else:
        print("  [Could not lay it — check the room name / date format.]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
