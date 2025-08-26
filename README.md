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

## 📊 Benchmark Results  

| Metric                  | Silero VAD                | TEN VAD                  |
|--------------------------|---------------------------|--------------------------|
| Speech segments          | 95                        | 41                       |
| Total speech             | 19.01s (13.0%)            | 114.98s (78.8%)          |
| Processing time          | 1.066s                    | 1.697s                   |
| Avg inference duration   | 0.22ms                    | 0.17ms                   |
| Min inference time       | 0.18ms                    | 0.14ms                   |
| Max inference duration   | 9.76ms                    | 0.78ms                   |

**Highlights:**  
- TEN VAD is **~1.27× faster per frame**  
- Detects **longer continuous speech** compared to Silero  
- Provides **lower latency** with fewer false segment splits  

## 🔧 Installation  

```bash
pip install livekit-plugins-tenvad
```

```bash
pip install git+https://github.com/dangvansam/livekit-plugins-tenvad.git
```