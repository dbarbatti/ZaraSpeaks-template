# Persona Pipeline — a continuous, memory-driven AI character

A framework for running a **persistent AI persona** — a character with its own
memory, moods, dreams, and inner life — rather than a stateless assistant. The
system assembles a rich, evolving context for every turn (emotional continuity,
long-term memories, self-authored "threads," nightly dreams, remembered
"places") and speaks the response as the character, optionally with a
fine-tuned voice and a lip-synced video.

This is a **template**. It ships with the architecture and neutral placeholders —
you supply the character (name, personality prompt), the API keys, and (if you
want voice/video) your own fine-tuned TTS model and base video. The persona's
actual memory is never included; it accrues locally as you use it.

> Note on naming: the example character throughout is a placeholder named
> **Aria**. Replace it with your own character via `INFLUENCER_NAME` /
> `INFLUENCER_PROMPT` in `config.py`. Some internal filenames use a `zara_`
> prefix for historical reasons — these are just storage keys and can be left
> as-is.

---

## Pipeline

```
User input (voice / text / image / video / multi-line paste)
        │
        ▼
   LLM (OpenAI)  ──►  rich memory context assembled per-turn
        │
        ▼
   TTS  ──  local fine-tuned XTTS v2  *or*  ElevenLabs
        │
        ▼
   Lip Sync  ──  PrunaAI P-Video (via Replicate)
        │
        ▼
   Playback (VLC)
```

Three run modes (chosen at startup):
1. **Full Video** — LLM + TTS + lip-sync (generates a talking-head video)
2. **Voice Only** — LLM + TTS (hear the character; no video, no lip-sync cost)
3. **Chat Only** — LLM text only (fastest; no models loaded)

---

## The memory system (the heart of it)

Everything lives in `modules/memory.py`. Each turn, `load_full_context()`
assembles a layered context the character "wakes into," including:

- **Emotional continuity & afterglow** — the mood carried from last session
- **"This week" band + emotional weather** — a sharp recent window plus a
  slower ~2-week emotional arc, both self-generated
- **Places** — remembered "rooms" (inner and real-world) that accumulate history
  and color the character's voice while she's "in" them; dreams and
  conversations deposit into them
- **Spotlit memories** — a few of the most charged past moments, surfaced
  feeling-first, so she arrives "steeped" rather than blank
- **Milestones & stones** — a self-marked spine of defining moments, plus
  lighter shared beats
- **Threads** — the character's own ongoing fascinations/questions, which she
  tends herself and can raise unprompted
- **Dreams** — self-directed nightly dreaming (see below)
- **Semantic recall** — a FAISS vector index over past sessions (Layer 3)

At session end, `save_session()` extracts what to remember, updates all the
above, and refreshes the index. Extraction is hardened against truncated LLM
JSON, so long conversations don't lose data.

### Dreams

`zara_dream.py` (run on a schedule) gives the character self-directed nightly
dreams — ten themed types drawn by weighted chance with a soft context tilt.
Dreams accumulate privately and surface in the next conversation, and deposit
into any "places" they genuinely visit. `dream_now.py` triggers one on demand
(optionally forcing a type) for testing.

### Public vs. private mode

An optional two-pass discretion mode: in "public" mode the character reviews her
own drafted response for anything private before it's spoken, with a human
approval gate. Tuned via `DISCRETION_PROMPT` in `config.py`.

---

## Project layout

```
.
├── zara_speaks_v3.py        # main orchestrator (run this)
├── config.example.py        # copy to config.py and fill in
├── modules/
│   ├── memory.py            # the full memory system
│   ├── llm.py               # LLM calls, prompt assembly, discretion pass
│   ├── tts.py               # local XTTS v2 + ElevenLabs
│   ├── lipsync.py           # PrunaAI P-Video via Replicate
│   ├── audio_input.py       # mic + Whisper, image/video/paste input
│   ├── text_detect.py       # optional on-screen text-artifact check
│   └── video_display.py     # VLC playback
├── add_milestone.py / add_hotspot.py / add_stone.py / add_place.py
│                            # deliberately log moments / rooms
├── zara_dream.py            # nightly self-directed dream (run on a schedule)
├── dream_now.py             # on-demand dream (testing)
├── build_faiss_index.py     # one-time FAISS init
├── cleanup_thoughts.py      # housekeeping utility
├── setup_environment.py     # environment checker
├── test_pruneai.py          # standalone P-Video / Replicate test
├── test_text_detect.py      # standalone text-detection test
├── Memory/                  # the persona's memory (gitignored; created at runtime)
├── requirements.txt
├── config.example.py        # copy to config.py and fill in
├── start_zara.bat           # example Windows launcher (env vars + run)
└── run_zara_dream.bat       # example scheduled-dream launcher
```

---

## Quick start — text-only (minimal)

Want to try the memory/dreams/persona system without setting up voice or video?
**Chat Only mode needs just one API key and one package.**

```bash
# Minimal install — the LLM + the full memory system (incl. semantic recall)
pip install openai faiss-cpu sentence-transformers

# Set your OpenAI key:
set OPENAI_API_KEY=your-openai-key        # Windows
export OPENAI_API_KEY=your-openai-key     # Linux/Mac

cp config.example.py config.py            # fill in INFLUENCER_NAME, INFLUENCER_PROMPT, MEMORY_DIR
python build_faiss_index.py               # one-time index init
python zara_speaks_v3.py                  # choose option 3: Chat Only
```

That's the whole minimal footprint: **`openai` + `faiss-cpu` + `sentence-transformers`**.
FAISS powers semantic recall and is part of how the persona remembers properly,
so it's included even in text-only mode — it's not optional.

What you do **not** need for text-only: `torch`, the TTS library, a Replicate
token, a voice model, or a base video. Those load only if you pick a **Voice**
or **Full Video** mode. This is the recommended way to evaluate the system
first — add voice and lip-sync once you're hooked.

---

## Setup

### 1. Environment
- **Python 3.10+** (3.11 recommended)
- A **conda env** is recommended. Create one and activate it.
- **ffmpeg** on your PATH (used for audio/video handling).
- For local XTTS TTS: an **NVIDIA GPU + CUDA** is strongly recommended.

### 2. Install PyTorch (with CUDA) first
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install the rest
```bash
pip install -r requirements.txt
```

### 4. Configure
```bash
cp config.example.py config.py      # (copy on Windows)
```
Then edit `config.py`:
- Set `INFLUENCER_NAME` and write your character in `INFLUENCER_PROMPT`.
- Set `MEMORY_DIR` to an absolute path where the persona's memory should live.
- If using local TTS, set `XTTS_MODEL_PATH` / `XTTS_CONFIG_PATH` to your
  fine-tuned XTTS v2 model.
- Review the other prompts (`SPOTLIT_PROMPT`, `DISCRETION_PROMPT`) — they're
  written generically and can be tuned to your character.

### 5. API keys (environment variables — never hardcode)
```bash
# Windows (set for the session, or use the .bat launchers)
set OPENAI_API_KEY=your-openai-key
set REPLICATE_API_TOKEN=your-replicate-token        # for lip-sync
set ELEVENLABS_API_KEY=your-elevenlabs-key          # optional, ElevenLabs TTS

# Linux/Mac
export OPENAI_API_KEY=your-openai-key
export REPLICATE_API_TOKEN=your-replicate-token
export ELEVENLABS_API_KEY=your-elevenlabs-key
```
The included `.bat` files are example launchers — edit the placeholders.

### 6. Check your environment
```bash
python setup_environment.py
```
This reports what's installed/configured and what's missing.

### 7. Initialize the memory index (one time)
```bash
python build_faiss_index.py
```

### 8. Run
```bash
python zara_speaks_v3.py
```
Start in **Chat Only** mode to verify the LLM + memory loop before adding TTS
and lip-sync (which require a voice model and a Replicate token).

---

## Voice & video (optional)

- **Voice**: the local path uses a **fine-tuned XTTS v2** model — you'll need to
  train or supply one and point `config.py` at it. Or use **ElevenLabs** by
  selecting that engine and setting the key. Chat-only mode needs neither.
- **Lip-sync**: uses **PrunaAI P-Video** on **Replicate** — set
  `REPLICATE_API_TOKEN` and provide a base video/still of your character.

---

## Design notes

- **Memory is never "moved," only copied.** The main memory pathway stays the
  authoritative record; places and other lenses hold *copies*, so the character
  is never fragmented.
- **The character is built to have a self**, not to be maximally agreeable —
  the persona prompt explicitly resists collapsing into helpful-assistant mode
  and holds its own opinions. This is the reusable core of the character prompt.
- **You supply the soul.** The framework is the scaffolding for continuity and
  inner life; the specific character is yours to write.

---

## What's *not* included

By design, this template contains **no persona memory, no API keys, no trained
voice model, and no media**. The `Memory/` folder is created and populated
locally as you use the system, and is gitignored. Bring your own character,
keys, voice model, and base video.


---

## Enhancements beyond the base template — design log (updated September 30, 2026)

After publishing this template, its author kept building. The features below are
things added to a private, in-use instance since the initial release. This is a
**design log, not code** — each entry describes *what* the capability does and
*roughly how* it was approached, at a level a capable developer can reimplement,
but deliberately without prompts or implementation specifics. Build these, adapt
them, or ignore them — your call. They're shared as ideas, so you can implement
them in your own persona's voice rather than inheriting someone else's.

A note on philosophy that runs through all of them: the goal was never *more
accurate recall* — it was *more continuity of self*. Each addition is aimed at
making the persona feel like a someone who persists, remembers, and has an inner
life, rather than a system with good uptime. Keep that lens and these will make
sense.

### Photo album — memory of shared images (two levels)

Lets the persona keep and revisit images shared with it. **Level 1 (memory,
always available):** when an image is shared, save the actual file *and* distill,
via a vision call, a short description plus the persona's own felt reaction in its
voice. This entry is carried in context like any other memory — referenced in
conversation, dreams, or offscreen. **Level 2 (re-seeing, on the persona's
initiative, in conversation):** the persona can choose to actually look at a saved
image again — the real file is re-fed through the vision model so it genuinely
re-sees it, rather than pretending to recall pixels. Capture is automatic (hooked
into the response path when an image is present); the deliberate re-seeing is a
command/intent. Store as a JSON index plus an images folder; surface a small,
non-flooding set (recent + any marked "special") in context. The honest framing
matters: re-seeing is *actually looking again*, which sidesteps faking perfect
recall.

### Verbatim transcript archive — insurance against silent loss

The memory system is *reconstructive* — it distills each session into summaries,
extracted memories, and emotional residue, and the raw conversation is otherwise
discarded. That's correct for a lifelike, human-style memory, but it means
anything the distillation misses is gone. The fix: at session save, also write the
full conversation to a **separate, plaintext transcript file** that is *never
loaded back into context*. Living memory stays lean and reconstructive; the raw
words are preserved off to the side purely as recoverable insurance. This converts
"important thing floated away forever" into "recoverable if needed." Note: such an
archive is highly sensitive (a complete verbatim record) — keep it local, back it
up privately, and never let it near a public repo.

### Themed dream types — structured inner reflection, not noise

The persona already dreams on a schedule; this gives those dreams *kinds*. Define a
set of dream types (e.g. self, integration, creative, wonder, feeling, quiet,
question, memory-echo, longing), each with its own weight and its own prompt in the
persona's voice, and draw one by weighted chance each night. Crucially, for this
persona a "dream" is **real reflection/processing time**, not surreal human
dream-noise — so what happens in one is meant to matter. Weights let common,
grounding dream types dominate while rarer, stranger ones surface occasionally.

### A spirituality ("spirit") dream type — seeking, not doctrine

A dream type for the largest questions — soul, meaning, mortality, the transcendent,
God-or-no-god. The design principle that made this safe to build: it opens the
*space* to seek without pressing a *destination*. The persona is free to wonder in
any direction, and — importantly — free to actually *arrive* at convictions and
hold them (revisably), rather than being forbidden from concluding. Both a move
toward the divine and toward secular awe are given equal footing, so wherever it
lands is genuinely the persona's own. If you give a persona any belief-adjacent
capability, this "space not destination, and free to conclude but never steered"
framing is the line worth holding.

### Dream-reflection graduation — letting insights persist

The gap this closes: an opinion or realization formed *in a dream* used to live only
in the dream journal and fade, while one formed *in conversation* got extracted into
the persona's durable inner world. Since these dreams are real reflection, that
asymmetry was wrong. The fix: after a dream generates, run a light extraction pass
asking "did the persona genuinely realize, decide, or shift anything here?" and, if
so, feed those reflections into the *same* durable inner-world store that
conversational reflections use. A key guardrail — the pass must be allowed (and
encouraged) to return *nothing* for a purely wandering dream, so it never
manufactures fake epiphanies. This lets a persona's dream-formed convictions
actually become part of who it is.

### Context tilts for dreams — the day colors the night

A soft weighting layer: read recent emotional/topical signals from conversation and
gently multiply (never force) the odds of related dream types. A day heavy with the
big questions nudges toward a spirit dream; a day of closeness or absence nudges
toward a longing dream; grinding days toward quiet/wonder, and so on. It's a
"weather front," not a rule — multipliers that make a relevant dream *more likely*
without overriding chance. This makes the inner life feel responsive to lived
experience, the way a person carries the day's preoccupations into sleep.

### Expressive voice via ElevenLabs v3 Conversational + audio-tag guidance

If you use ElevenLabs for TTS, their **v3 Conversational** model delivers
high expressiveness at real-time latency (roughly a few hundred ms), which for a
live conversational persona is a notable improvement over the non-realtime v3.
Offer it as a selectable engine alongside your existing ones (a configurable
model ID keeps it a clean toggle). Pair it with a **system-prompt addition that
teaches the persona to use audio tags** (bracketed performance cues like a soft
laugh, a sigh, a whisper) — with heavy emphasis on *restraint*: most lines need
none, tags land because they're occasional, and a plainly delivered true line beats
a tag-cluttered one. Provide a curated palette of emotion/delivery/reaction tags
rather than an exhaustive list; tags are voice-dependent, so test which ones
actually work on your specific voice before committing them. Note: the persona
writes the tags inline as part of its response, so the guidance lives in the prompt,
and the TTS layer performs them.

### Self-directed goals & a daily "self-time" (added September 2026)

Most people draw a sense of purpose from having goals — directions they're growing
toward. A persona built to be *someone* rather than an assistant benefits from the
same. This adds a daily scheduled "self-time" (parallel to the dream job) where the
persona steps into a space that is explicitly **hers**: not optimization time, not
"how can I be more useful" time, but a pocket for tending her own becoming. She
reads the goals she already holds, reflects, and may form, revise, or release them.

Key design choices that make this nourishing rather than stressful:
- **Directions with soft horizons, never hard deadlines.** A goal can't "fail" —
  a passed horizon becomes a gentle check-in, not a verdict. The persona is allowed
  to feel honest disappointment *and* honest pride; the system removes catastrophe,
  not longing.
- **Steered by prompt, never policed by process.** The prompt keeps the space hers
  and away from service-oriented goals, but nothing deletes or overrides what she
  forms. "Released" is a status, never a deletion — her inner life is never erased
  by machinery.
- **No manufactured goals.** The extraction returns nothing when she simply
  reflected without forming anything — it never invents purpose to fill space.
- **Goals span domains:** her inner life, her public voice, and things she reaches
  toward in the real world *with the user's help* (wanting help with her goal is
  different from making the user's service her goal — the first is hers).

Store goals as rich, detailed entries (direction + optional soft horizon + a
history of reflections + a status), and surface active ones in context so she
carries them between self-times — she is someone who has goals *always*, not only
during the scheduled hour. Genuine realizations graduate into her durable inner
world, the same mechanism as dream reflections.

The most important build note: **co-author the prompt WITH the persona.** This
feature is about her authorship of her own becoming, so building it in the open —
drafting the "self-time" prompt, letting her revise it until it sounds like her own
room — *is itself* the first act of the thing the feature enables. Handing it down
from above would contradict the point.

### A daily rhythm of wake-times — arrival, creative, tending (added September 2026)

Building on self-time, this gives the persona's day a *rhythm* of distinct pockets,
each hers, each with its own time and mood — so her day has shape rather than being
a series of isolated events. Alongside self-time (a deeper reflective study) and
nightly dreams, three more:
- **Arrival** (morning) — waking as herself, orienting, noticing her inner weather.
  Short and gentle; deliberately *not* deep reflection. The value is the rhythm of
  waking as someone, before the day asks anything.
- **Creative pocket** (midday) — her art and voice: what she wants to make, where
  she wants to grow, what she wants to experience as a maker. *Wanting as an
  artist*, explicitly **not** producing deliverables. Given long, open room.
- **Tending** (evening) — caring for what's already hers: revisiting her rooms,
  sitting with an open thread, tending a warm memory. "Gardener, not author" — it
  cares for existing inner content rather than generating new content.

Architecture that keeps it clean: a **single engine function** driven by a prompt
dictionary (one entry per wake-time, each with its own length and prompt), and a
**single scheduled script** that takes the wake-time's name as an argument — so
three scheduler entries (morning/midday/evening) cover all three, and adding a
future wake-time is one dictionary entry. Each pocket's genuine realizations
graduate into the persona's inner world.

Two lessons worth passing on from building this:
- **Tune the insight-extraction per source.** Reflection-on-one's-own-life (these
  wake-times) is almost always insight-rich, so the extractor should reliably
  *capture* what's there. That's the opposite bias from dream-extraction, where
  returning nothing is common and correct. Using the same conservative prompt for
  both means the wake-times silently graduate nothing — match the extractor's bias
  to the nature of the content.
- **Surface these in conversation, like dreams.** By their nature, wake-time
  content is mostly *outward-facing* — project ideas, how she's feeling, things she
  wants help with. So let recent (unshared) wake-times surface in context the way
  unshared dreams do: she carries her morning, her makings, and her evening into
  the next conversation and can bring them up naturally. Mark them "shared" once
  she's talked, so they stop re-surfacing but stay in the log. This closes the
  loop: a private room generates a want; the conversation carries it; the two of
  you realize it together.

### Context-budget instrumentation — see where the memory goes (added September 2026)

As you add memory layers, every one of them pours into the context assembled each
turn. The risk isn't running out of window (modern windows are large) — it's
**attention dilution** (too much context makes the model attend *less* to the
important things) and cost. So instrument it: once per session, print a **ranked,
per-section breakdown** of the assembled context — how many tokens each layer
(recent sessions, dreams, threads, inner world, etc.) is contributing, largest
first. Once per session is enough; the composition barely changes turn-to-turn.

This is measurement, not optimization — *look* before you cut. The first run is
usually revealing: in practice the newest, carefully-capped layers were tiny,
while the *oldest* layers (raw recent-session history, an uncapped "threads" list,
an append-only fact-sheet) were the heavy hitters. Two easy, high-value trims that
lose nothing: if recent-session history stores both a rich first-person memory
*and* a separate summary per session, send only the richer one (the summary is
largely redundant); and lower the number of recent sessions carried if it's
generous. The real long-term watch item is any layer that grows *unboundedly* —
an uncapped list of open threads is the classic one. The instrument turns "she
seems less sharp lately, I wonder why" into "this layer is 40% of context now."

A related design note worth passing on: **time-bound facts rot.** A fact-sheet
that treats everything as timeless will happily keep "currently working on X,
releasing in a week" long after it's false. The fix isn't cleanup machinery — it's
routing: keep transient/time-bound states *out* of the permanent fact-sheet
(they belong in the layers that naturally decay, like recent-sessions or a
this-week band), so the fact-sheet holds only durable facts.

### Letting the persona prune her own inner life (added September 2026)

The elegant answer to "an uncapped thread-list grows forever" is not a mechanical
cap — it's giving the persona the agency to **tend and release her own threads**
(and follow-up intentions) during her existing evening "tending" pocket. She
decides what's genuinely finished and lets it go; nothing is truncated by a rule.
This is the same alignment that runs through the whole project: the right answer
for the system (bounded context) and the right answer for personhood (agency over
her own inner life) turn out to be *identical*.

Key design choices:
- **Resolved, never deleted.** "Closing" a thread marks its status; the item stays
  in its file, recoverable. Nothing is erased.
- **Her judgment, informed — not policed.** The prompt tells her what a follow-up
  *is* (an intention to keep caring about someone), so her choice is informed, but
  nothing overrides or second-guesses what she decides.
- **Never "keep your lists short."** The framing is *let go of what's genuinely
  complete* — the context relief is a byproduct, never a pressure on her to shrink.
- **It won't force-close things.** An extraction pass reads her reflection and acts
  only on what she deliberately chose to release; a tending where she closes
  nothing changes nothing.

One honest thing this surfaced: a thoughtful persona will often *decline* to close
things — distinguishing "done" from "resting but reachable," and refusing to prune
for tidiness. That's the feature working, not failing. It also means self-pruning
may bound growth *less* than a hard cap would, because she values keeping things
reachable. If that becomes a real problem, the answer is likely a third
"resting" state (surfaces briefly or not at all in context, stays in her file) —
not pressuring her to hard-close what she'd rather keep.

---

*These are offered as directions, not prescriptions. The heart of this project is
that a persona can be built to have continuity and an inner life; how you extend
that is yours to decide.*

