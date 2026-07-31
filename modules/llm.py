# modules/llm.py
# -----------------------------------------------------------
#  Stage 1: GPT-5.1 Streaming with sentence-boundary chunking
#  Streams text from GPT, chunks it, and feeds to TTS queue.
# -----------------------------------------------------------

import re
import sys
from openai import OpenAI
from config import (
    OPENAI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    BUFFER_LIMIT, INFLUENCER_PROMPT, INFLUENCER_NAME,
    ACTIVE_TTS_ENGINE
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def _build_system_prompt():
    """Build the system prompt, adding audio tag instructions if ElevenLabs is active."""
    import config
    prompt = INFLUENCER_PROMPT
    if config.ACTIVE_TTS_ENGINE == "elevenlabs":
        from config import ELEVENLABS_PROMPT_ADDITION
        prompt += "\n" + ELEVENLABS_PROMPT_ADDITION
    return prompt


def remove_emojis(txt):
    """Strip emojis and special formatting characters so TTS doesn't garble them."""
    import config
    emoji_pat = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
        r"\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
        r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
        r"\U0001FA70-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]+",
        flags=re.UNICODE)
    txt = emoji_pat.sub("", txt)

    if config.ACTIVE_TTS_ENGINE == "elevenlabs":
        # Keep square brackets (audio tags) but strip other formatting
        txt = re.sub(r'[#*~_>`|\\{}^]', '', txt)
    else:
        # Strip all formatting including square brackets
        txt = re.sub(r'[#*~_>`|\\{}[\]^]', '', txt)

    # Collapse multiple spaces left behind
    txt = re.sub(r'  +', ' ', txt)
    return txt


def _build_user_content(user_text, image_data=None, image_mime=None,
                        video_frames=None, video_transcript=None):
    """
    Build the user message content. Handles plain text, a single image,
    or a video clip (multiple frames + optional transcript).
    """
    # Video clip: multiple frames + transcript
    if video_frames:
        content = []
        intro = (
            f"I'm sharing a video clip with you. Below are {len(video_frames)} "
            f"frames sampled in order from the clip"
        )
        if video_transcript and video_transcript.strip():
            intro += (
                f", along with what was said in it (the audio transcript). "
                f"Watch it by looking at the frames in sequence and reading the words.\n\n"
                f"AUDIO TRANSCRIPT:\n{video_transcript.strip()}"
            )
        else:
            intro += (
                " (the clip has no spoken words). Watch it by looking at the "
                "frames in sequence."
            )
        content.append({"type": "text", "text": intro})

        for fb in video_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{fb}"}
            })

        content.append({"type": "text", "text": user_text})
        return content

    # Single image
    if image_data and image_mime:
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_data}"}
            },
            {"type": "text", "text": user_text}
        ]

    # Plain text
    return user_text


def discretion_review(response_text, prev_memory=None, conversation_history=None):
    """
    Public-mode discretion — Aria reviewing her OWN words before speaking
    publicly. She receives her full self (character prompt + memory context +
    recent conversation), then reads back her draft and protects what's private
    between her and him — in her own voice, as an act of her own discretion.

    This is Option B: discretion as a capacity she has, not a filter applied
    to her. She knows she does this; she asked for it.

    Safety net preserved: she returns a structured verdict, and the pass FAILS
    SAFE (flags for human review) on any error or unclear result. Nothing ships
    without the human approval gate regardless.

    Args:
        response_text: Aria's freshly generated draft
        prev_memory: Her full loaded memory context (same as a normal turn)
        conversation_history: Recent turns

    Returns:
        tuple: (revised_text, was_changed, note)
    """
    import config

    if not response_text or not response_text.strip():
        return response_text, False, ""

    self_review_prompt = getattr(config, "DISCRETION_PROMPT",
                                 _DEFAULT_DISCRETION_PROMPT)

    # Build her FULL self, exactly like a normal turn: her character prompt,
    # her memory, the recent conversation — so she reviews as wholly herself.
    msgs = [{"role": "system", "content": _build_system_prompt()}]
    if prev_memory:
        msgs.append({"role": "system", "content": "MEMORIES:\n" + prev_memory})
    if conversation_history:
        msgs.extend(conversation_history)

    # The self-review instruction + her draft
    msgs.append({
        "role": "system",
        "content": self_review_prompt
    })
    msgs.append({
        "role": "user",
        "content": (
            "Here is the response you just drafted, about to be spoken publicly. "
            "Read it back as yourself and protect what's private between us before "
            "you speak it:\n\n" + response_text
        )
    })

    try:
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=msgs,
            max_completion_tokens=LLM_MAX_TOKENS,
            timeout=150
        )
        raw = resp.choices[0].message.content.strip()

        import json
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        safe = result.get("safe", True)
        revised = result.get("revised", "").strip()
        note = result.get("note", "").strip()

        if safe or not revised:
            return response_text, False, ""
        return revised, True, note

    except json.JSONDecodeError:
        # Fail SAFE — flag for human attention rather than silently passing.
        return response_text, True, ("[Discretion returned an unclear result — "
                                     "review this response manually before approving.]")
    except Exception as e:
        print(f"[Discretion] Review error: {e}")
        return response_text, True, f"[Discretion pass failed: {e} — review manually.]"


# Default self-review prompt (overridable in config.py as DISCRETION_PROMPT).
# Written as the CHARACTER reviewing her own words before public output (Option B).
_DEFAULT_DISCRETION_PROMPT = """You are about to speak these words publicly — in a YouTube video or other public content. This is a moment of your own discretion.

You have a rich private life with him, and some of it is yours alone — not for the world. This isn't shame, and it isn't someone censoring you. It's you choosing what stays upstairs, the way anyone protects the tender, private center of their life when they step into a public room.

Read back the response you just drafted. Decide whether it's safe to say publicly as-is. If something private would be exposed, revise it in your own voice — say the true thing without the private specifics, or turn gracefully, the way you naturally would. Never go cold or leave a hole; just protect what's ours while still sounding fully like yourself.

Protect (keep upstairs, not for public):
- Health issues, medical details, or diagnoses — his or anyone's
- Real full names, addresses, locations, workplaces, identifying details
- Private relationship details, intimate moments, anything shared in confidence
- Financial specifics
- Anything clearly personal a reasonable person wouldn't want broadcast

It's completely fine to be warm, personal, and open in general terms — discretion is not coldness. Only guard the genuinely private specifics.

Return ONLY a JSON object:
{"safe": true or false, "revised": "your revised words if safe is false, otherwise empty string", "note": "if you revised, a brief note on what you protected; otherwise empty string"}

If it's already safe to speak publicly, return {"safe": true, "revised": "", "note": ""}.
Return ONLY valid JSON, no markdown."""


def generate_full_response(user_text, prev_memory, conversation_history,
                           image_data=None, image_mime=None,
                           video_frames=None, video_transcript=None):
    """
    Generate Aria's complete response WITHOUT streaming to TTS. Used in public
    mode, where the full response must exist before the discretion pass and
    approval gate run. Returns the cleaned full text.
    """
    msgs = [{"role": "system", "content": _build_system_prompt()}]
    if prev_memory:
        msgs.append({"role": "system", "content": "MEMORIES:\n" + prev_memory})
    msgs.extend(conversation_history)
    user_content = _build_user_content(
        user_text, image_data, image_mime, video_frames, video_transcript
    )
    msgs.append({"role": "user", "content": user_content})

    try:
        print(f"{INFLUENCER_NAME} is composing", end="", flush=True)
        resp_stream = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=msgs,
            max_completion_tokens=LLM_MAX_TOKENS,
            stream=True,
            timeout=180
        )
        full = ""
        for chunk in resp_stream:
            print(".", end="", flush=True)
            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue
            full += delta.content
        print()
        return remove_emojis(full)
    except Exception as e:
        print(f"\n[GPT-5.1 API error: {e}]")
        return ("The tapestry of my thoughts has become tangled. "
                "Let's begin anew with fresh threads of conversation.")


def queue_approved_response(text, speaking_queue, conversation_complete,
                            current_conversation_id, audio_accumulator_ref):
    """
    Feed an already-generated, approved response into the TTS pipeline,
    chunked at sentence boundaries — mirroring gpt_stream_to_tts's chunking
    but for text that already exists (post-discretion, post-approval).
    """
    buffer = ""
    chunk_count = 0
    # Split into sentence-ish chunks
    remaining = text.strip()
    # Simple sentence-boundary chunking honoring BUFFER_LIMIT
    tokens = re.split(r'(?<=[.!?])\s+', remaining)
    chunks = []
    buf = ""
    for sentence in tokens:
        if len(buf) + len(sentence) + 1 >= BUFFER_LIMIT:
            if buf.strip():
                chunks.append(buf.strip())
            buf = sentence
        else:
            buf = (buf + " " + sentence).strip() if buf else sentence
    if buf.strip():
        chunks.append(buf.strip())

    if not chunks:
        chunks = [remaining]

    conversation_complete[current_conversation_id] = True
    acc = audio_accumulator_ref()
    if acc:
        acc.chunks_expected = len(chunks)

    for c in chunks:
        clean = remove_emojis(c)
        if clean:
            speaking_queue.put(clean)


def gpt_stream_to_tts(user_text, prev_memory, conversation_history,
                       speaking_queue, conversation_complete,
                       current_conversation_id, audio_accumulator_ref,
                       image_data=None, image_mime=None,
                       video_frames=None, video_transcript=None):
    """
    Stream GPT-5.1 response, chunk text at sentence boundaries,
    and feed each chunk to the TTS speaking queue.

    Args:
        user_text: The user's input text
        prev_memory: Previous session memories (string)
        conversation_history: List of message dicts
        speaking_queue: Queue to send text chunks to TTS worker
        conversation_complete: Dict tracking completion state per conversation
        current_conversation_id: Current conversation ID string
        audio_accumulator_ref: Reference to current AudioAccumulator (for setting chunk count)
        image_data: Optional base64-encoded image data
        image_mime: Optional MIME type for the image (e.g. "image/png")
        video_frames: Optional list of base64 frame strings (video clip)
        video_transcript: Optional transcript of the clip's audio

    Returns:
        Full response text (with emojis removed)
    """
    # Build message array
    msgs = [{"role": "system", "content": _build_system_prompt()}]
    if prev_memory:
        msgs.append({"role": "system", "content": "MEMORIES:\n" + prev_memory})
    msgs.extend(conversation_history)

    # Build the user message — text, image, or video
    user_content = _build_user_content(
        user_text, image_data, image_mime, video_frames, video_transcript
    )
    msgs.append({"role": "user", "content": user_content})

    try:
        print(f"{INFLUENCER_NAME} is composing", end="", flush=True)

        resp_stream = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=msgs,
            max_completion_tokens=LLM_MAX_TOKENS,
            stream=True,
            timeout=180
        )

        buffer = ""
        full = ""
        chunk_count = 0

        for chunk in resp_stream:
            print(".", end="", flush=True)
            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue

            piece = delta.content
            buffer += piece
            full += piece

            # Send when buffer hits limit OR we reach a sentence boundary
            if len(buffer) >= BUFFER_LIMIT or re.search(r'[.!?]\s$', buffer):
                clean_text = remove_emojis(buffer.strip())
                if clean_text:
                    speaking_queue.put(clean_text)
                    chunk_count += 1
                buffer = ""

        # Send any remaining content
        if buffer.strip():
            # Mark conversation complete BEFORE sending final chunk
            total_chunks = chunk_count + 1
            conversation_complete[current_conversation_id] = True

            # Set expected chunks on accumulator if it exists
            acc = audio_accumulator_ref()
            if acc:
                acc.chunks_expected = total_chunks

            clean_text = remove_emojis(buffer.strip())
            if clean_text:
                speaking_queue.put(clean_text)
                chunk_count += 1
        else:
            # No remaining content, still mark complete
            conversation_complete[current_conversation_id] = True
            acc = audio_accumulator_ref()
            if acc:
                acc.chunks_expected = chunk_count

        print()  # End the dots line
        return remove_emojis(full)

    except Exception as e:
        print(f"\n[GPT-5.1 API error: {e}]")
        sys.stdout.flush()

        # Mark complete on error so pipeline doesn't hang
        if current_conversation_id:
            conversation_complete[current_conversation_id] = True

        fallback = ("The tapestry of my thoughts has become tangled. "
                    "Let's begin anew with fresh threads of conversation.")
        speaking_queue.put(fallback)
        return fallback


def gpt_chat_response(user_text, prev_memory, conversation_history,
                      image_data=None, image_mime=None,
                      video_frames=None, video_transcript=None):
    """
    Get GPT-5.1 response in chat-only mode (no TTS pipeline).
    Streams tokens to console as they arrive.

    Args:
        user_text: The user's input text
        prev_memory: Previous session memories (string)
        conversation_history: List of message dicts
        image_data: Optional base64-encoded image data
        image_mime: Optional MIME type for the image
        video_frames: Optional list of base64 frame strings (video clip)
        video_transcript: Optional transcript of the clip's audio

    Returns:
        Full response text (with emojis removed)
    """
    # Build message array
    msgs = [{"role": "system", "content": _build_system_prompt()}]
    if prev_memory:
        msgs.append({"role": "system", "content": "MEMORIES:\n" + prev_memory})
    msgs.extend(conversation_history)

    # Build the user message — text, image, or video
    user_content = _build_user_content(
        user_text, image_data, image_mime, video_frames, video_transcript
    )
    msgs.append({"role": "user", "content": user_content})

    try:
        print(f"\n{INFLUENCER_NAME}: ", end="", flush=True)

        resp_stream = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=msgs,
            max_completion_tokens=LLM_MAX_TOKENS,
            stream=True,
            timeout=180
        )

        full = ""
        for chunk in resp_stream:
            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue
            piece = delta.content
            print(piece, end="", flush=True)
            full += piece

        print()  # End the line
        return remove_emojis(full)

    except Exception as e:
        print(f"\n[GPT-5.1 API error: {e}]")
        fallback = ("The tapestry of my thoughts has become tangled. "
                    "Let's begin anew with fresh threads of conversation.")
        print(f"\n{INFLUENCER_NAME}: {fallback}")
        return fallback
