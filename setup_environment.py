"""
setup_environment.py — verify the environment for the current pipeline:
    LLM (OpenAI) -> XTTS v2 (local) or ElevenLabs -> PrunaAI P-Video (Replicate)

Run:  python setup_environment.py
Reports what's installed/configured and what still needs attention.
Nothing here is destructive — it only checks and reports.
"""
import os
import sys
import importlib


def status(name, ok, detail=""):
    mark = "OK " if ok else "-- "
    line = f"[{mark}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def check_python():
    v = sys.version_info
    return status("Python 3.10+", v >= (3, 10), f"found {v.major}.{v.minor}")


def check_package(display, import_name=None):
    mod = import_name or display
    try:
        importlib.import_module(mod)
        return status(display, True)
    except Exception:
        return status(display, False, f"pip install {display}")


def check_cuda():
    try:
        import torch
        ok = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if ok else "no CUDA GPU (local XTTS will be slow/CPU)"
        return status("PyTorch + CUDA", ok, detail)
    except Exception:
        return status("PyTorch", False, "install torch with CUDA (see requirements.txt)")


def check_ffmpeg():
    from shutil import which
    return status("ffmpeg", which("ffmpeg") is not None, "install ffmpeg and put it on PATH")


def check_env_keys():
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    status("OPENAI_API_KEY", openai_ok, "required — set as env var")
    repl_ok = bool(os.getenv("REPLICATE_API_TOKEN"))
    status("REPLICATE_API_TOKEN", repl_ok, "required for lip-sync (PrunaAI P-Video)")
    el_ok = bool(os.getenv("ELEVENLABS_API_KEY"))
    status("ELEVENLABS_API_KEY", el_ok, "optional — only for ElevenLabs TTS mode")
    return openai_ok and repl_ok


def check_config():
    try:
        import config  # noqa
        return status("config.py present", True, "copied from config.example.py")
    except Exception:
        return status("config.py present", False, "copy config.example.py -> config.py and fill it in")


def check_dirs():
    ok = True
    for d in ("Memory", "voice_references", "base_videos"):
        exists = os.path.isdir(d)
        status(f"dir: {d}/", exists, "" if exists else "will be created / add your assets here")
        ok = ok and exists or d != "Memory"  # only Memory is strictly required
    return ok


def main():
    print("=" * 60)
    print("  Environment check — LLM -> XTTS/ElevenLabs -> P-Video pipeline")
    print("=" * 60)
    checks = [
        check_python(),
        check_config(),
        check_cuda(),
        check_ffmpeg(),
        check_package("openai"),
        check_package("replicate"),
        check_package("TTS"),
        check_package("faiss", "faiss"),
        check_package("sentence-transformers", "sentence_transformers"),
        check_package("whisper"),
        check_env_keys(),
        check_dirs(),
    ]
    print("=" * 60)
    if all(checks):
        print("  All core checks passed. Initialize FAISS then run the app:")
    else:
        print("  Some checks need attention (see above). Then:")
    print("    python build_faiss_index.py      # one-time FAISS init")
    print("    python zara_speaks_v3.py         # start the app")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
