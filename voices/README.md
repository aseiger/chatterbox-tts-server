Voice reference clips for the /tts endpoint.

Add a voice:
  1. Drop a 5-15s clean speech clip here, e.g.  voices/emma.wav
     (wav/mp3/ogg/flac — anything soundfile can read)
  2. Request it by name:  {"text": "...", "reference_audio_filename": "emma"}
     (or "emma.wav"; unknown/empty names fall back to the built-in voice)

No server restart needed.
