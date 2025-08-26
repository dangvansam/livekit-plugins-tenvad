import asyncio
import os
import shutil
import sys
import wave
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from livekit import rtc
from livekit.agents import vad
from livekit.plugins import silero, tenvad


@dataclass
class VADConfig:
    """Configuration for a VAD model"""

    name: str
    model_type: str
    min_speech_duration: float = 0.1
    min_silence_duration: float = 0.15
    activation_threshold: float = 0.5
    sample_rate: int = 16000
    chunk_duration_ms: int = 32
    color: str = "blue"


def read_wav_file(file_path: str) -> Tuple[np.ndarray, int, int]:
    """Read a WAV file and return audio data, sample rate, and number of channels"""
    with wave.open(file_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()

        # Read raw audio data
        raw_data = wav_file.readframes(num_frames)

        # Convert to numpy array
        if wav_file.getsampwidth() == 2:  # 16-bit audio
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
        elif wav_file.getsampwidth() == 1:  # 8-bit audio
            audio_data = np.frombuffer(raw_data, dtype=np.uint8)
            audio_data = (audio_data.astype(np.int16) - 128) * 256
        else:
            raise ValueError(f"Unsupported sample width: {wav_file.getsampwidth()}")

        # Handle multi-channel audio (convert to mono)
        if num_channels > 1:
            audio_data = audio_data.reshape(-1, num_channels)
            audio_data = np.mean(audio_data, axis=1).astype(np.int16)
            num_channels = 1

        return audio_data, sample_rate, num_channels


def write_wav_file(file_path: str, audio_data: np.ndarray, sample_rate: int, num_channels: int = 1):
    """Write audio data to a WAV file"""
    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


def resample_audio(audio_data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Simple resampling using linear interpolation"""
    if orig_rate == target_rate:
        return audio_data

    # Calculate the resampling ratio
    ratio = target_rate / orig_rate
    new_length = int(len(audio_data) * ratio)

    # Create indices for interpolation
    old_indices = np.arange(len(audio_data))
    new_indices = np.linspace(0, len(audio_data) - 1, new_length)

    # Perform linear interpolation
    resampled = np.interp(new_indices, old_indices, audio_data)

    return resampled.astype(np.int16)


def extract_segment(
    audio_data: np.ndarray, sample_rate: int, start: float, end: float
) -> np.ndarray:
    """Extract audio segment based on timestamps"""
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)

    return audio_data[start_sample:end_sample]


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60
    milliseconds = int((secs % 1) * 1000)
    secs = int(secs)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def save_vad_outputs(
    vad_results: dict,
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: Path,
    save_segments: bool = True,
    save_srt: bool = True,
    save_combined: bool = True,
) -> None:
    """Save VAD outputs (segments, SRT, combined audio) for a single model"""
    vad_name = vad_results["name"]
    segments = vad_results["speech_segments"]

    if not segments:
        print(f"  No speech segments found for {vad_name}")
        return

    # Create output directory for this VAD
    vad_dir = output_path / f"{vad_name.lower()}_segments"
    vad_dir.mkdir(exist_ok=True)

    # Save individual segments
    if save_segments:
        print(f"  Saving {len(segments)} {vad_name} segments to {vad_dir}/")
        for i, seg in enumerate(segments, 1):
            segment_audio = extract_segment(audio_data, sample_rate, seg["start"], seg["end"])
            segment_file = vad_dir / f"segment_{i:03d}_{seg['start']:.2f}s_{seg['end']:.2f}s.wav"
            write_wav_file(str(segment_file), segment_audio, sample_rate)

    # Save SRT file
    if save_srt:
        srt_file = output_path / f"{vad_name.lower()}_subtitles.srt"
        save_srt_file(segments, str(srt_file), vad_name)
        print(f"  {vad_name} SRT file saved: {srt_file}")

    # Save combined speech-only file
    if save_combined:
        combined_segments = []
        for seg in segments:
            segment_audio = extract_segment(audio_data, sample_rate, seg["start"], seg["end"])
            combined_segments.append(segment_audio)
            # Add small silence between segments
            combined_segments.append(np.zeros(int(sample_rate * 0.1), dtype=np.int16))

        if combined_segments:
            combined_audio = np.concatenate(combined_segments)
            combined_file = output_path / f"{vad_name.lower()}_speech_only.wav"
            write_wav_file(str(combined_file), combined_audio, sample_rate)
            print(f"  {vad_name} combined speech file: {combined_file}")


def save_srt_file(segments: list, output_path: str, model_name: str = ""):
    """Save segments as SRT subtitle file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")

            duration = seg["end"] - seg["start"]
            if model_name:
                f.write(f"[{model_name} Speech {i}: {duration:.2f}s]\n")
            else:
                f.write(f"[Speech {i}: {duration:.2f}s]\n")

            f.write("\n")


def calculate_agreement(results1: dict, results2: dict, duration: float) -> float:
    """Calculate agreement percentage between two VAD results"""
    # Create binary arrays for comparison
    resolution = 100  # samples per second
    time_bins = int(duration * resolution)
    silero_binary = np.zeros(time_bins)
    ten_binary = np.zeros(time_bins)

    # Fill binary arrays
    for seg in results1["speech_segments"]:
        start_idx = int(seg["start"] * resolution)
        end_idx = min(int(seg["end"] * resolution), time_bins)
        silero_binary[start_idx:end_idx] = 1

    for seg in results2["speech_segments"]:
        start_idx = int(seg["start"] * resolution)
        end_idx = min(int(seg["end"] * resolution), time_bins)
        ten_binary[start_idx:end_idx] = 1

    # Calculate agreement
    agreement = np.mean(silero_binary == ten_binary) * 100
    return agreement


def create_comparison_visualization(
    audio_data: np.ndarray,
    sample_rate: int,
    silero_threshold: float,
    silero_results: dict,
    ten_threshold: float,
    ten_results: dict,
    output_path: Path,
    input_file: str,
):
    """Create comparison visualization of Silero and TEN VAD results"""

    # Calculate time axis
    duration = len(audio_data) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_data))

    # Create figure with subplots (adding 2 more for inference times)
    fig, axes = plt.subplots(8, 1, figsize=(16, 18), sharex=True)
    fig.suptitle(f"VAD Comparison: {os.path.basename(input_file)}", fontsize=16, fontweight="bold")

    # 1. Waveform
    ax1 = axes[0]
    ax1.plot(time_axis, audio_data, color="gray", alpha=0.6, linewidth=0.5)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title("Audio Waveform", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Silero VAD probability
    ax2 = axes[1]
    if silero_results["inference_events"]:
        timestamps = [e["timestamp"] for e in silero_results["inference_events"]]
        probabilities = [e["probability"] for e in silero_results["inference_events"]]

        ax2.plot(timestamps, probabilities, color="blue", linewidth=1.5, label="Silero")
        ax2.fill_between(timestamps, probabilities, alpha=0.3, color="blue")
        ax2.axhline(y=silero_threshold, color="red", linestyle="--", alpha=0.5, label="Threshold")

    ax2.set_ylabel("Probability", fontsize=10)
    ax2.set_title("Silero VAD - Speech Probability", fontsize=11)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # 3. TEN VAD probability
    ax3 = axes[2]
    if ten_results["inference_events"]:
        timestamps = [e["timestamp"] for e in ten_results["inference_events"]]
        probabilities = [e["probability"] for e in ten_results["inference_events"]]

        ax3.plot(timestamps, probabilities, color="green", linewidth=1.5, label="TEN")
        ax3.fill_between(timestamps, probabilities, alpha=0.3, color="green")
        ax3.axhline(y=ten_threshold, color="red", linestyle="--", alpha=0.5, label="Threshold")

    ax3.set_ylabel("Probability", fontsize=10)
    ax3.set_title("TEN VAD - Speech Probability", fontsize=11)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    # 4. Silero segments timeline
    ax4 = axes[3]
    ax4.set_title(
        f"Silero VAD - {silero_results['num_segments']} segments, {silero_results['speech_ratio']:.1f}% speech",
        fontsize=11,
    )
    ax4.set_ylabel("Silero", fontsize=10)

    segment_height = 0.8
    for seg in silero_results["speech_segments"]:
        ax4.barh(
            0,
            seg["duration"],
            left=seg["start"],
            height=segment_height,
            color="blue",
            alpha=0.6,
            edgecolor="darkblue",
            linewidth=1,
        )

    ax4.set_ylim([-0.5, 0.5])
    ax4.set_yticks([])
    ax4.grid(True, alpha=0.3, axis="x")

    # 5. TEN segments timeline
    ax5 = axes[4]
    ax5.set_title(
        f"TEN VAD - {ten_results['num_segments']} segments, {ten_results['speech_ratio']:.1f}% speech",
        fontsize=11,
    )
    ax5.set_ylabel("TEN", fontsize=10)

    for seg in ten_results["speech_segments"]:
        ax5.barh(
            0,
            seg["duration"],
            left=seg["start"],
            height=segment_height,
            color="green",
            alpha=0.6,
            edgecolor="darkgreen",
            linewidth=1,
        )

    ax5.set_ylim([-0.5, 0.5])
    ax5.set_yticks([])
    ax5.grid(True, alpha=0.3, axis="x")

    # 6. Difference visualization
    ax6 = axes[5]
    ax6.set_title("Agreement and Disagreement Regions", fontsize=11)
    ax6.set_xlabel("Time (seconds)", fontsize=10)
    ax6.set_ylabel("Agreement", fontsize=10)

    # Create binary arrays for comparison
    resolution = 100  # samples per second
    time_bins = np.linspace(0, duration, int(duration * resolution))
    silero_binary = np.zeros(len(time_bins))
    ten_binary = np.zeros(len(time_bins))

    # Fill binary arrays based on segments
    for seg in silero_results["speech_segments"]:
        start_idx = int(seg["start"] * resolution)
        end_idx = int(seg["end"] * resolution)
        silero_binary[start_idx:end_idx] = 1

    for seg in ten_results["speech_segments"]:
        start_idx = int(seg["start"] * resolution)
        end_idx = int(seg["end"] * resolution)
        ten_binary[start_idx:end_idx] = 1

    # Calculate agreement/disagreement
    both_speech = (silero_binary == 1) & (ten_binary == 1)
    both_silence = (silero_binary == 0) & (ten_binary == 0)
    only_silero = (silero_binary == 1) & (ten_binary == 0)
    only_ten = (silero_binary == 0) & (ten_binary == 1)

    # Plot agreement/disagreement regions
    ax6.fill_between(
        time_bins, 0, both_speech * 0.5, color="purple", alpha=0.5, label="Both detect speech"
    )
    ax6.fill_between(
        time_bins, 0, both_silence * (-0.5), color="gray", alpha=0.3, label="Both detect silence"
    )
    ax6.fill_between(time_bins, 0, only_silero * 0.25, color="blue", alpha=0.5, label="Only Silero")
    ax6.fill_between(time_bins, 0, only_ten * (-0.25), color="green", alpha=0.5, label="Only TEN")

    ax6.set_ylim([-0.6, 0.6])
    ax6.set_xlim([0, duration])
    ax6.axhline(y=0, color="black", linewidth=0.5)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc="upper right", ncol=2)

    # 7. Inference time comparison
    ax7 = axes[6]
    ax7.set_title("Inference Time per Frame", fontsize=11)
    ax7.set_ylabel("Time (ms)", fontsize=10)

    # Plot Silero inference times
    if silero_results.get("inference_events"):
        silero_timestamps = []
        silero_inference_ms = []
        for event in silero_results["inference_events"]:
            if event.get("inference_duration") is not None:
                silero_timestamps.append(event["timestamp"])
                silero_inference_ms.append(event["inference_duration"] * 1000)

        if silero_inference_ms:
            ax7.plot(
                silero_timestamps,
                silero_inference_ms,
                color="blue",
                alpha=0.5,
                linewidth=0.8,
                label=f"Silero (avg: {np.mean(silero_inference_ms):.2f}ms)",
            )

    # Plot TEN inference times
    if ten_results.get("inference_events"):
        ten_timestamps = []
        ten_inference_ms = []
        for event in ten_results["inference_events"]:
            if event.get("inference_duration") is not None:
                ten_timestamps.append(event["timestamp"])
                ten_inference_ms.append(event["inference_duration"] * 1000)

        if ten_inference_ms:
            ax7.plot(
                ten_timestamps,
                ten_inference_ms,
                color="green",
                alpha=0.5,
                linewidth=0.8,
                label=f"TEN (avg: {np.mean(ten_inference_ms):.2f}ms)",
            )

    ax7.grid(True, alpha=0.3)
    ax7.legend(loc="upper right")

    # 8. Rolling average inference time comparison
    ax8 = axes[7]
    ax8.set_title("Inference Time Comparison (10-frame Rolling Average)", fontsize=11)
    ax8.set_ylabel("Time (ms)", fontsize=10)
    ax8.set_xlabel("Time (seconds)", fontsize=10)

    # Plot rolling averages
    window_size = 10

    if silero_results.get("inference_events") and silero_inference_ms:
        if len(silero_inference_ms) >= window_size:
            silero_rolling = np.convolve(
                silero_inference_ms, np.ones(window_size) / window_size, mode="valid"
            )
            silero_rolling_timestamps = silero_timestamps[: len(silero_rolling)]
            ax8.plot(
                silero_rolling_timestamps, silero_rolling, color="blue", linewidth=2, label="Silero"
            )
            # Add average line
            ax8.axhline(y=np.mean(silero_inference_ms), color="blue", linestyle="--", alpha=0.3)

    if ten_results.get("inference_events") and ten_inference_ms:
        if len(ten_inference_ms) >= window_size:
            ten_rolling = np.convolve(
                ten_inference_ms, np.ones(window_size) / window_size, mode="valid"
            )
            ten_rolling_timestamps = ten_timestamps[: len(ten_rolling)]
            ax8.plot(ten_rolling_timestamps, ten_rolling, color="green", linewidth=2, label="TEN")
            # Add average line
            ax8.axhline(y=np.mean(ten_inference_ms), color="green", linestyle="--", alpha=0.3)

    ax8.grid(True, alpha=0.3)
    ax8.legend(loc="upper right")

    # Adjust layout and save
    plt.tight_layout()

    # Save the figure
    plot_file = output_path / "vad_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_file


def create_vad_instance(config: VADConfig):
    """Create a VAD instance based on configuration"""
    if config.model_type == "silero":
        return silero.VAD.load(
            min_speech_duration=config.min_speech_duration,
            min_silence_duration=config.min_silence_duration,
            activation_threshold=config.activation_threshold,
            sample_rate=config.sample_rate,
        )
    elif config.model_type == "ten":
        return tenvad.VAD.load(
            min_speech_duration=config.min_speech_duration,
            min_silence_duration=config.min_silence_duration,
            activation_threshold=config.activation_threshold,
            sample_rate=config.sample_rate,
        )
    else:
        raise ValueError(f"Unknown VAD model type: {config.model_type}")


async def process_with_vad(
    audio_data: np.ndarray,
    sample_rate: int,
    vad_instance,
    config: VADConfig,
) -> dict:
    """Process audio with a given VAD instance and return results"""

    print(f"Processing with {config.name}...")
    import time

    start_time = time.time()
    stream = vad_instance.stream()
    chunk_size = int(sample_rate * config.chunk_duration_ms / 1000)
    frames = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i : i + chunk_size]
        if len(chunk) >= chunk_size:
            frame = rtc.AudioFrame(
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=chunk_size,
                data=chunk.tobytes(),
            )
            frames.append(frame)
            stream.push_frame(frame)

    stream.end_input()
    logger.success(f"Pushed {len(frames)} frames to stream VAD: {config.name}")

    speech_segments = []
    current_segment = None
    inference_events = []
    inference_durations = []
    count = 0
    start_time = time.time()
    async for event in stream:
        if event.type == vad.VADEventType.START_OF_SPEECH:
            current_segment = {
                "start": round(event.timestamp, 4),
                "start_index": event.samples_index,
            }

        elif event.type == vad.VADEventType.END_OF_SPEECH:
            if current_segment:
                current_segment["end"] = round(event.timestamp, 4)
                current_segment["end_index"] = event.samples_index
                current_segment["duration"] = round(event.timestamp - current_segment["start"], 4)
                speech_segments.append(current_segment)
                print(current_segment)
                current_segment = None

        elif event.type == vad.VADEventType.INFERENCE_DONE:
            inference_event_data = {
                "timestamp": event.timestamp,
                "probability": event.probability,
                "is_speech": event.speaking,
                "inference_duration": event.inference_duration
                if hasattr(event, "inference_duration")
                else None,
            }
            inference_events.append(inference_event_data)

            if current_segment:
                current_segment["is_speech"] = event.speaking

            if hasattr(event, "inference_duration") and event.inference_duration:
                inference_durations.append(event.inference_duration)

        count += 1

    logger.success(f"Processed {count} inference events from stream VAD: {config.name}")
    processing_time = time.time() - start_time

    # Calculate inference statistics
    avg_inference_duration = np.mean(inference_durations) if inference_durations else 0
    max_inference_duration = np.max(inference_durations) if inference_durations else 0
    min_inference_time = np.min(inference_durations) if inference_durations else 0

    # Calculate statistics
    total_duration = len(audio_data) / sample_rate
    total_speech_duration = sum(seg["duration"] for seg in speech_segments)
    speech_ratio = (total_speech_duration / total_duration * 100) if total_duration > 0 else 0

    return {
        "name": config.name,
        "speech_segments": speech_segments,
        "inference_events": inference_events,
        "total_speech_duration": total_speech_duration,
        "speech_ratio": speech_ratio,
        "num_segments": len(speech_segments),
        "processing_time": processing_time,
        "avg_inference_duration": avg_inference_duration,
        "max_inference_duration": max_inference_duration,
        "min_inference_time": min_inference_time,
        "num_inferences": len(inference_durations),
        "config": config,
    }


async def process_audio_file(
    input_file: str, output_dir: str = None, vad_configs: Optional[List[VADConfig]] = None
):
    """Process audio file with one or more VAD models"""

    # Read the input audio file
    print(f"Reading audio file: {input_file}")
    audio_data, original_sample_rate, num_channels = read_wav_file(input_file)

    print(f"Audio info: {original_sample_rate}Hz, {num_channels} channel(s), {len(audio_data)} samples")
    print(f"Duration: {len(audio_data) / original_sample_rate:.2f} seconds")

    # TEN VAD supports 8kHz and 16kHz, so resample if needed
    target_sample_rate = 16000
    if original_sample_rate not in [16000]:
        print(f"Resampling from {original_sample_rate}Hz to {target_sample_rate}Hz...")
        audio_data = resample_audio(audio_data, original_sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    else:
        sample_rate = original_sample_rate
        target_sample_rate = sample_rate

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(".")

    if vad_configs is None:
        print("No VAD configurations provided")
        exit()

    # Initialize and process with VAD models
    print("Initializing VAD models...")
    vad_results = []

    for config in vad_configs:
        config.sample_rate = target_sample_rate
        vad_instance = create_vad_instance(config)
        results = await process_with_vad(audio_data, sample_rate, vad_instance, config)
        vad_results.append(results)

    # Calculate comparison metrics
    total_duration = len(audio_data) / sample_rate

    print("" + "=" * 60)
    print("VAD RESULTS")
    print("=" * 60)

    # Display results for each VAD
    for result in vad_results:
        print(f"{result['name']} VAD:")
        print(f"  - Speech segments: {result['num_segments']}")
        print(
            f"  - Total speech: {result['total_speech_duration']:.2f}s ({result['speech_ratio']:.1f}%)"
        )
        print(f"  - Processing time: {result['processing_time']:.3f}s")
        if result["num_inferences"] > 0:
            print(f"  - Avg inference duration: {result['avg_inference_duration'] * 1000:.2f}ms")
            print(f"  - Min inference time: {result['min_inference_time'] * 1000:.2f}ms")
            print(f"  - Max inference duration: {result['max_inference_duration'] * 1000:.2f}ms")

    # Compare VADs if there are exactly 2
    if len(vad_results) == 2:
        print("Comparison Metrics:")
        agreement = calculate_agreement(vad_results[0], vad_results[1], total_duration)
        print(f"  - Agreement: {agreement:.1f}%")
        print(
            f"  - Speech duration difference: {abs(vad_results[0]['total_speech_duration'] - vad_results[1]['total_speech_duration']):.2f}s"
        )
        print(
            f"  - Segment count difference: {abs(vad_results[0]['num_segments'] - vad_results[1]['num_segments'])}"
        )

        # Compare based on inference duration if available
        if vad_results[0]["num_inferences"] > 0 and vad_results[1]["num_inferences"] > 0:
            inference_speed_ratio = (
                vad_results[1]["avg_inference_duration"] / vad_results[0]["avg_inference_duration"]
            )
            faster_model = (
                vad_results[0]["name"] if inference_speed_ratio > 1 else vad_results[1]["name"]
            )
            slower_model = (
                vad_results[1]["name"] if inference_speed_ratio > 1 else vad_results[0]["name"]
            )
            ratio = max(inference_speed_ratio, 1 / inference_speed_ratio)
            print(
                f"  - Inference speed: {faster_model} is {ratio:.2f}x faster than {slower_model} (per frame)"
            )

    # Save outputs for each VAD
    if output_dir:
        print("Saving VAD outputs...")
        for result in vad_results:
            save_vad_outputs(
                result,
                audio_data,
                sample_rate,
                output_path,
                save_segments=True,
                save_srt=True,
                save_combined=True,
            )

    # Save comparison report
    if output_dir:
        report_file = output_path / "comparison_report.txt"
        with open(report_file, "w") as f:
            f.write("VAD Report\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Duration: {total_duration:.2f}s\n")
            f.write(f"Sample rate: {sample_rate}Hz\n\n")

            # Write results for each VAD
            for result in vad_results:
                f.write(f"{result['name']} VAD Results:\n")
                f.write(f"  Speech segments: {result['num_segments']}\n")
                f.write(
                    f"  Total speech: {result['total_speech_duration']:.2f}s ({result['speech_ratio']:.1f}%)\n"
                )
                f.write(f"  Processing time: {result['processing_time']:.3f}s\n")
                if result["num_inferences"] > 0:
                    f.write(
                        f"  Avg inference duration: {result['avg_inference_duration'] * 1000:.2f}ms\n"
                    )
                    f.write(f"  Min inference time: {result['min_inference_time'] * 1000:.2f}ms\n")
                    f.write(
                        f"  Max inference duration: {result['max_inference_duration'] * 1000:.2f}ms\n"
                    )
                    f.write(f"  Number of inferences: {result['num_inferences']}\n")
                f.write("\n")

            # Write segment details
            f.write("Segment Details:\n")
            f.write(f"{'-' * 40}\n")
            for result in vad_results:
                f.write(f"{result['name']} segments:\n")
                for i, seg in enumerate(result["speech_segments"], 1):
                    f.write(
                        f"  {i}: {seg['start']:.2f}s - {seg['end']:.2f}s (duration: {seg['duration']:.2f}s)\n"
                    )
                f.write("\n")

        print(f"Comparison report saved to: {report_file}")

    # Create comparison visualization (only if exactly 2 VADs)
    if output_dir and len(vad_results) == 2:
        print("Creating comparison visualization with inference time analysis...")
        plot_file = create_comparison_visualization(
            audio_data=audio_data,
            sample_rate=sample_rate,
            ten_threshold=vad_results[1]["config"].activation_threshold
            if vad_results[1]["name"] == "TEN"
            else vad_results[0]["config"].activation_threshold,
            silero_threshold=vad_results[0]["config"].activation_threshold
            if vad_results[0]["name"] == "Silero"
            else vad_results[1]["config"].activation_threshold,
            silero_results=vad_results[0] if vad_results[0]["name"] == "Silero" else vad_results[1],
            ten_results=vad_results[1] if vad_results[1]["name"] == "TEN" else vad_results[0],
            output_path=output_path,
            input_file=input_file,
        )
        print(f"Visualization saved to: {plot_file}")

    return vad_results


def main():
    if len(sys.argv) < 2:
        print("Usage: python test/benchmark.py <input_audio_file> [output_dir] [vad_models]")
        print("Examples:")
        print("  python test/benchmark.py sample.wav                    # Compare all VADs")
        print("  python test/benchmark.py sample.wav outputs/            # Specify output dir")
        print("  python test/benchmark.py sample.wav outputs/ silero     # Use only Silero VAD")
        print("  python test/benchmark.py sample.wav outputs/ ten        # Use only TEN VAD")
        print("  python test/benchmark.py sample.wav outputs/ silero,ten # Compare specific VADs")
        print("Supported VAD models: silero, ten")
        print("Supported formats: WAV files (will be resampled to 16kHz if needed)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "vad_output"

    # Parse VAD models selection
    vad_configs = None
    if len(sys.argv) > 3:
        vad_models = sys.argv[3].lower().split(",")
        vad_configs = []

        for model in vad_models:
            model = model.strip()
            if model == "silero":
                vad_configs.append(
                    VADConfig(
                        name="Silero",
                        model_type="silero",
                        min_speech_duration=0.15,
                        min_silence_duration=0.05,
                        activation_threshold=0.5,
                        sample_rate=16000,
                        chunk_duration_ms=32,
                        color="blue",
                    )
                )
            elif model == "ten":
                vad_configs.append(
                    VADConfig(
                        name="TEN",
                        model_type="ten",
                        min_speech_duration=0.15,
                        min_silence_duration=0.05,
                        activation_threshold=0.5,
                        sample_rate=16000,
                        chunk_duration_ms=16,
                        color="green",
                    )
                )
            else:
                print(f"Error: Unknown VAD model '{model}'")
                print("Supported models: silero, ten")
                sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists. Removing...")
        shutil.rmtree(output_dir)

    # Run the async processing
    asyncio.run(process_audio_file(input_file, output_dir, vad_configs))
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
