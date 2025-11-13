# LiveKit Plugins – TEN VAD  

**`livekit-plugins-tenvad`** provides seamless integration of the [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad) voice activity detection (VAD) plugin into the [LiveKit](https://github.com/livekit) ecosystem.  

This plugin enables **real-time speech activity detection** with low-latency inference, optimized for streaming, conversational AI, and **[livekit-agents](https://github.com/livekit/agents)** integration.  

## ✨ Features  
- 🔌 **LiveKit plugin integration** — plug-and-play support for LiveKit workflows  
- 🤖 **Compatible with livekit-agents** — extend agents with real-time VAD capabilities  
- 🎤 **Accurate voice activity detection** powered by [TEN VAD](https://github.com/TEN-framework/ten-vad)  
- ⚡ **Low-latency inference** (~0.17ms avg per frame) suitable for real-time use  
- 📊 **Benchmark validated** against Silero VAD (faster and more continuous speech detection)  
- 🛠️ **Configurable & extensible** within the LiveKit plugin system  


## 🔧 Installation  
```bash
# from PyPI
uv pip install livekit-plugins-tenvad

# from source
uv pip install git+https://github.com/dangvansam/livekit-plugins-tenvad.git
```
## 🔌 Usage
```python
from livekit.plugins import tenvad

vad = tenvad.VAD.load(
    activation_threshold=0.5,
    min_speech_duration=0.1,
    min_silence_duration=0.3,
    max_buffered_speech=60,
    prefix_padding_duration=0.3
)
```

## 📊 Run Benchmark  
```bash
git clone https://github.com/dangvansam/livekit-plugins-tenvad.git

cd livekit-plugins-tenvad

# install dependencies for testing
uv pip install .[test]

python test/benchmark.py -i test/sample.wav
```


### Benchmark Results  

| Metric                  | Silero VAD                | TEN VAD                  |
|--------------------------|---------------------------|--------------------------|
| Speech segments          | 95                        | 41                       |
| Total speech             | 19.01s (13.0%)            | 114.98s (78.8%)          |
| Avg inference time       | 0.22ms                    | 0.17ms                   |
| Min inference time       | 0.18ms                    | 0.14ms                   |
| Max inference time       | 9.76ms                    | 0.78ms                   |

**Highlights:**  
- TEN VAD is **~1.27× faster per frame**  
- Detects **longer continuous speech** compared to Silero  
- Provides **lower latency** with fewer false segment splits  

## Visualizations
### Long audio
![TEN VAD Benchmark](test/vad_comparison_long_audio.png)

### Short audio
![TEN VAD Benchmark](test/vad_comparison_short_audio.png)

## Citations
```
@misc{TEN VAD,
  author       = {TEN Team},
  title        = {TEN VAD: A Low-Latency, Lightweight and High-Performance Streaming Voice Activity Detector (VAD)},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/TEN-framework/ten-vad.git}},
  email        = {developer@ten.ai}
}
```
