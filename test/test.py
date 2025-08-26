from livekit.plugins import tenvad

vad=tenvad.VAD.load(
    activation_threshold=0.5,
    min_speech_duration=0.2,
    min_silence_duration=0.2,
    sample_rate=16000,
)