# modules/tts.py
# -----------------------------------------------------------
#  Stage 2: Local TTS via XTTS v2
#  Zero-shot voice cloning from reference WAV.
#  Replaces ElevenLabs cloud API.
# -----------------------------------------------------------

import os
import sys
import time
import glob
import wave
import struct
import numpy as np
from config import (
    TTS_LANGUAGE, REFERENCE_WAV_DIR,
    TTS_TEMPERATURE, TTS_REPETITION_PENALTY, TTS_TOP_K, TTS_TOP_P,
    TTS_SAMPLE_RATE, TTS_ENABLE_DEEPSPEED, TEMP_DIR, DEBUG,
    PRESERVE_AUDIO, AUDIO_LIBRARY_DIR
, XTTS_MODEL_PATH, XTTS_CONFIG_PATH)
# Fix for PyTorch 2.6+ weights_only default change
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


class LocalTTS:
    """
    XTTS v2 local text-to-speech with zero-shot voice cloning.

    Phase 1: Zero-shot — uses a reference WAV to clone voice.
    Phase 2: Fine-tuned — swap model weights, same interface.
    """

    def __init__(self):
        self.model = None
        self.speaker_wav = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self._loaded = False

    def load(self, reference_wav=None):
        """
        Load XTTS v2 model and compute speaker embedding from reference WAV.
        This should be called once at startup — model stays in VRAM.

        Args:
            reference_wav: Path to reference WAV file. If None, auto-selects
                          from REFERENCE_WAV_DIR.
        """
        print("[TTS] Loading XTTS v2 model...")
        start = time.time()

        try:
            from TTS.api import TTS as CoquiTTS

            # Load the model
            #### default model
#            self.model = CoquiTTS(
#                model_name=TTS_MODEL_NAME,
#                gpu=True
#            )
            # XTTS_MODEL_PATH / XTTS_CONFIG_PATH come from config.py — point them
            # at your own fine-tuned XTTS v2 model directory and its config.json.
            self.model = CoquiTTS(
                model_path=XTTS_MODEL_PATH,
                config_path=XTTS_CONFIG_PATH,
                gpu=True
            )

            # Select reference WAV
            if reference_wav and os.path.exists(reference_wav):
                self.speaker_wav = reference_wav
            else:
                self.speaker_wav = self._find_reference_wav()

            if not self.speaker_wav:
                raise FileNotFoundError(
                    f"No reference WAV found in {REFERENCE_WAV_DIR}\n"
                    "Please add a 10-30 second WAV of Aria's voice."
                )

            print(f"[TTS] Using reference voice: {os.path.basename(self.speaker_wav)}")

            # Pre-compute speaker conditioning for faster inference
            # This extracts the voice characteristics once
            self._precompute_speaker_embedding()

            elapsed = time.time() - start
            print(f"[TTS] Model loaded in {elapsed:.1f}s")
            self._loaded = True
            return True

        except Exception as e:
            print(f"[TTS] ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
 

    def _find_reference_wav(self):
        """Find a reference WAV file in the voice_references directory."""
        os.makedirs(REFERENCE_WAV_DIR, exist_ok=True)
        patterns = ["*.wav", "*.WAV"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(REFERENCE_WAV_DIR, p)))

        if not files:
            return None

        # Use the first one found (user can specify explicitly)
        return files[0]

    def _precompute_speaker_embedding(self):
        """
        Pre-compute GPT conditioning latent and speaker embedding.
        This avoids re-processing the reference WAV on every TTS call.
        """
        try:
            # Access the underlying model for direct inference
            tts_model = self.model.synthesizer.tts_model

            gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
                audio_path=[self.speaker_wav],
                gpt_cond_len=30,    # Use up to 30 seconds of reference
                gpt_cond_chunk_len=6
            )

            self.gpt_cond_latent = gpt_cond_latent
            self.speaker_embedding = speaker_embedding
            print("[TTS] Speaker embedding computed and cached")

        except Exception as e:
            print(f"[TTS] WARNING: Could not pre-compute embedding: {e}")
            print("[TTS] Will use per-call reference (slower)")
            self.gpt_cond_latent = None
            self.speaker_embedding = None

    def synthesize(self, text):
        """
        Convert text to speech audio bytes (WAV format).

        Args:
            text: String to synthesize

        Returns:
            bytes: Raw WAV file content, or None on failure
        """
        if not self._loaded:
            print("[TTS] ERROR: Model not loaded. Call load() first.")
            return None

        if not text or not text.strip():
            return None

        try:
            start = time.time()

            if self.gpt_cond_latent is not None:
                # Fast path: use pre-computed embeddings
                tts_model = self.model.synthesizer.tts_model
                result = tts_model.inference(
                    text=text,
                    language=TTS_LANGUAGE,
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                    temperature=TTS_TEMPERATURE,
                    repetition_penalty=TTS_REPETITION_PENALTY,
                    top_k=TTS_TOP_K,
                    top_p=TTS_TOP_P,
                    enable_text_splitting=True
                )
                # result["wav"] is a numpy array of float32 samples
                audio_array = result["wav"]
            else:
                # Slow path: re-process reference WAV each time
                temp_out = os.path.join(TEMP_DIR, f"tts_temp_{int(time.time())}.wav")
                self.model.tts_to_file(
                    text=text,
                    file_path=temp_out,
                    speaker_wav=self.speaker_wav,
                    language=TTS_LANGUAGE
                )
                # Read back the file
                with open(temp_out, 'rb') as f:
                    wav_bytes = f.read()
                try:
                    os.remove(temp_out)
                except OSError:
                    pass
                elapsed = time.time() - start
                if DEBUG:
                    print(f"[TTS] Generated {len(wav_bytes)} bytes in {elapsed:.2f}s (slow path)")
                return wav_bytes

            # Convert float32 numpy array to WAV bytes
            wav_bytes = self._array_to_wav_bytes(audio_array)

            elapsed = time.time() - start
            duration = len(audio_array) / TTS_SAMPLE_RATE
            rtf = elapsed / duration if duration > 0 else 0

            if DEBUG:
                print(f"[TTS] Generated {duration:.1f}s audio in {elapsed:.2f}s "
                      f"(RTF: {rtf:.2f}x)")

            return wav_bytes

        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

    def _array_to_wav_bytes(self, audio_array):
        """Convert a numpy float32 audio array to WAV file bytes."""
        import io

        # Normalize to int16 range
        if isinstance(audio_array, np.ndarray):
            audio_data = audio_array
        else:
            # Handle torch tensor
            audio_data = np.array(audio_array)

        # Ensure float32 and clip
        audio_data = audio_data.astype(np.float32)
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Write WAV to bytes buffer
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(TTS_SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        return buf.getvalue()

    def list_reference_voices(self):
        """List available reference WAV files."""
        os.makedirs(REFERENCE_WAV_DIR, exist_ok=True)
        files = []
        for ext in ["*.wav", "*.WAV"]:
            files.extend(glob.glob(os.path.join(REFERENCE_WAV_DIR, ext)))
        return files

    def set_reference_voice(self, wav_path):
        """
        Switch to a different reference voice.
        Re-computes speaker embedding.
        """
        if not os.path.exists(wav_path):
            print(f"[TTS] Reference file not found: {wav_path}")
            return False

        self.speaker_wav = wav_path
        print(f"[TTS] Switching to reference: {os.path.basename(wav_path)}")
        self._precompute_speaker_embedding()
        return True


class ElevenLabsTTS:
    """
    ElevenLabs v3 text-to-speech with audio tag support.

    Supports emotion and non-verbal tags like [laughs], [excited], [whispers].
    Requires ELEVENLABS_API_KEY environment variable and elevenlabs Python SDK.
    """

    def __init__(self):
        self.client = None
        self._loaded = False

    def load(self):
        """Initialize the ElevenLabs client."""
        from config import (ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
                            ELEVENLABS_MODEL_ID)

        if not ELEVENLABS_API_KEY:
            print("[ElevenLabs TTS] ERROR: ELEVENLABS_API_KEY not set")
            print("[Set it with: set ELEVENLABS_API_KEY=your-key-here]")
            return False

        if not ELEVENLABS_VOICE_ID:
            print("[ElevenLabs TTS] ERROR: ELEVENLABS_VOICE_ID not set in config.py")
            print("[Get your voice ID from the ElevenLabs voice library]")
            return False

        try:
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            print(f"[ElevenLabs TTS] Client initialized")
            print(f"[ElevenLabs TTS] Model: {ELEVENLABS_MODEL_ID}")
            print(f"[ElevenLabs TTS] Voice ID: {ELEVENLABS_VOICE_ID}")
            self._loaded = True
            return True
        except ImportError:
            print("[ElevenLabs TTS] ERROR: elevenlabs SDK not installed")
            print("[Install with: pip install elevenlabs]")
            return False
        except Exception as e:
            print(f"[ElevenLabs TTS] ERROR: {e}")
            return False

    def synthesize(self, text):
        """
        Convert text to speech via ElevenLabs v3 API.
        Supports audio tags like [laughs], [excited], etc.

        Args:
            text: String to synthesize (may include audio tags)

        Returns:
            bytes: WAV file content, or None on failure
        """
        if not self._loaded:
            print("[ElevenLabs TTS] ERROR: Client not loaded. Call load() first.")
            return None

        if not text or not text.strip():
            return None

        from config import (ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID,
                            ELEVENLABS_STABILITY, ELEVENLABS_SIMILARITY_BOOST,
                            ELEVENLABS_OUTPUT_FORMAT)

        try:
            start = time.time()

            # Generate audio via ElevenLabs API
            audio_iterator = self.client.text_to_speech.convert(
                text=text,
                voice_id=ELEVENLABS_VOICE_ID,
                model_id=ELEVENLABS_MODEL_ID,
                output_format=ELEVENLABS_OUTPUT_FORMAT,
                voice_settings={
                    "stability": ELEVENLABS_STABILITY,
                    "similarity_boost": ELEVENLABS_SIMILARITY_BOOST,
                }
            )

            # Collect MP3 bytes from iterator
            mp3_bytes = b"".join(audio_iterator)

            if not mp3_bytes:
                print("[ElevenLabs TTS] No audio returned")
                return None

            # Convert MP3 to WAV for pipeline compatibility
            wav_bytes = self._mp3_to_wav(mp3_bytes)

            elapsed = time.time() - start
            if DEBUG:
                print(f"[ElevenLabs TTS] Generated {len(mp3_bytes)} bytes "
                      f"in {elapsed:.2f}s")

            return wav_bytes

        except Exception as e:
            print(f"[ElevenLabs TTS] Synthesis error: {e}")
            return None

    def _mp3_to_wav(self, mp3_bytes):
        """Convert MP3 bytes to WAV bytes using ffmpeg."""
        import subprocess
        import tempfile

        # Write MP3 to temp file
        mp3_path = os.path.join(TEMP_DIR, f"el_temp_{int(time.time())}.mp3")
        wav_path = os.path.join(TEMP_DIR, f"el_temp_{int(time.time())}.wav")

        try:
            with open(mp3_path, 'wb') as f:
                f.write(mp3_bytes)

            # Convert to WAV matching pipeline sample rate
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_path,
                 "-ar", str(TTS_SAMPLE_RATE), "-ac", "1",
                 "-c:a", "pcm_s16le", wav_path],
                capture_output=True, text=True
            )

            if result.returncode == 0 and os.path.exists(wav_path):
                with open(wav_path, 'rb') as f:
                    wav_bytes = f.read()
                return wav_bytes
            else:
                print(f"[ElevenLabs TTS] FFmpeg conversion failed: {result.stderr[:200]}")
                return None

        finally:
            try:
                os.remove(mp3_path)
            except OSError:
                pass
            try:
                os.remove(wav_path)
            except OSError:
                pass


class AudioAccumulator:
    """
    Accumulates audio chunks from TTS until the full conversation
    response is complete, then combines into a single WAV file.

    Same pattern as V2 — just faster since TTS is local.
    """

    def __init__(self, conversation_id):
        self.conversation_id = conversation_id
        self.audio_segments = []   # List of WAV bytes
        self.chunks_received = 0
        self.chunks_expected = None

    def add_audio_chunk(self, wav_bytes):
        """
        Add a WAV audio chunk. Returns True when all chunks are in
        and we should send to lip sync.
        """
        self.audio_segments.append(wav_bytes)
        self.chunks_received += 1

        if DEBUG:
            print(f"[Accumulator] Chunk {self.chunks_received}: "
                  f"{len(wav_bytes)} bytes received")

        return self.is_complete()

    def is_complete(self):
        """Check if we have all expected chunks."""
        return (
            self.chunks_expected is not None
            and self.chunks_received >= self.chunks_expected
        )

    def get_combined_wav(self):
        """
        Combine all WAV segments into a single WAV file.
        Handles WAV headers properly — strips headers from individual
        segments and writes one unified header.
        """
        if not self.audio_segments:
            return None

        try:
            import io
            all_pcm = bytearray()
            sample_rate = TTS_SAMPLE_RATE
            sample_width = 2  # 16-bit
            channels = 1

            for wav_bytes in self.audio_segments:
                # Parse WAV to extract raw PCM data
                buf = io.BytesIO(wav_bytes)
                try:
                    with wave.open(buf, 'rb') as wf:
                        sample_rate = wf.getframerate()
                        sample_width = wf.getsampwidth()
                        channels = wf.getnchannels()
                        pcm_data = wf.readframes(wf.getnframes())
                        all_pcm.extend(pcm_data)
                except wave.Error:
                    # If it's not a valid WAV, try treating as raw PCM
                    all_pcm.extend(wav_bytes)

            # Write combined WAV
            out_buf = io.BytesIO()
            with wave.open(out_buf, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(bytes(all_pcm))

            combined = out_buf.getvalue()
            total_duration = len(all_pcm) / (sample_rate * sample_width * channels)
            print(f"[Accumulator] Combined {len(self.audio_segments)} segments: "
                  f"{total_duration:.1f}s total audio")

            return combined

        except Exception as e:
            print(f"[Accumulator] Error combining audio: {e}")
            return None


def speaker_worker(speaking_queue, audio_file_queue, conversation_complete,
                   get_conversation_id, get_accumulator, set_accumulator,
                   tts_engine):
    """
    TTS worker thread. Reads text chunks from speaking_queue,
    generates audio locally via XTTS v2, accumulates until
    conversation is complete, then sends combined WAV to lip sync.

    Args:
        speaking_queue: Input queue of text chunks
        audio_file_queue: Output queue for complete WAV file paths
        conversation_complete: Dict of {conv_id: bool}
        get_conversation_id: Callable returning current conversation ID
        get_accumulator: Callable returning current AudioAccumulator
        set_accumulator: Callable to set/clear the AudioAccumulator
        tts_engine: Loaded LocalTTS instance
    """
    print("[TTS Worker] Started")
    sys.stdout.flush()

    while True:
        text = speaking_queue.get()
        if text is None:
            print("[TTS Worker] Shutting down")
            break

        try:
            conv_id = get_conversation_id()

            # Create accumulator if needed
            acc = get_accumulator()
            if acc is None or acc.conversation_id != conv_id:
                acc = AudioAccumulator(conv_id)
                set_accumulator(acc)

            # Generate audio locally
            wav_bytes = tts_engine.synthesize(text)

            if wav_bytes:
                should_send = acc.add_audio_chunk(wav_bytes)

                # Also check if queue is empty and conversation is marked complete
                if (speaking_queue.empty()
                        and conversation_complete.get(conv_id, False)
                        and not should_send):
                    acc.chunks_expected = acc.chunks_received
                    should_send = True

                if should_send:
                    combined_wav = acc.get_combined_wav()
                    if combined_wav:
                        # Write to temp file for lip sync worker
                        temp_path = os.path.join(
                            TEMP_DIR,
                            f"speech_complete_{conv_id}_{int(time.time())}.wav"
                        )
                        with open(temp_path, 'wb') as f:
                            f.write(combined_wav)

                        # Preserve audio to library if enabled
                        if PRESERVE_AUDIO:
                            try:
                                from datetime import datetime
                                os.makedirs(AUDIO_LIBRARY_DIR, exist_ok=True)
                                audio_name = datetime.now().strftime(
                                    "zara_audio_%Y%m%d_%H%M%S.wav"
                                )
                                library_path = os.path.join(
                                    AUDIO_LIBRARY_DIR, audio_name
                                )
                                import shutil
                                shutil.copy2(temp_path, library_path)
                                print(f"[TTS Worker] Audio saved: {library_path}")
                            except Exception as e:
                                print(f"[TTS Worker] Audio save warning: {e}")

                        audio_file_queue.put(temp_path)
                        print(f"[TTS Worker] Complete audio ready: {temp_path}")

                        # Clear accumulator
                        set_accumulator(None)
            else:
                print("[TTS Worker] WARNING: TTS returned no audio")

        except Exception as e:
            print(f"[TTS Worker] Error: {e}")
        finally:
            speaking_queue.task_done()
