import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter, wiener
import noisereduce as nr
from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SR = 16000


# --------------------------------------------------
# BANDPASS FILTER (Speech Frequency Filter)
# --------------------------------------------------

def bandpass_filter(data, lowcut=300, highcut=3400, fs=16000, order=5):

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    filtered = lfilter(b, a, data)

    return filtered


# --------------------------------------------------
# EXTRACT AUDIO FROM VIDEO
# --------------------------------------------------

def extract_audio(video_path, audio_path):

    video = VideoFileClip(str(video_path))

    video.audio.write_audiofile(
        audio_path,
        fps=SR,
        codec="pcm_s16le",
        verbose=False,
        logger=None
    )


# --------------------------------------------------
# AUDIO NOISE SUPPRESSION PIPELINE
# --------------------------------------------------

def clean_audio(input_audio, output_audio):

    print("Loading audio...")

    audio, sr = librosa.load(input_audio, sr=SR)

    # Step 1: Speech band filtering
    print("Applying bandpass filter...")
    audio = bandpass_filter(audio, 300, 3400, sr)

    # Step 2: Spectral noise reduction
    print("Applying spectral noise reduction...")
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=1.0
    )

    # Step 3: Wiener smoothing filter
    print("Applying Wiener filter...")
    audio = wiener(audio)

    print("Saving cleaned audio...")
    sf.write(output_audio, audio, sr)


# --------------------------------------------------
# MERGE CLEAN AUDIO BACK TO VIDEO
# --------------------------------------------------

def merge_audio_video(video_path, clean_audio_path, output_video):

    print("Merging clean audio with video...")

    video = VideoFileClip(str(video_path))
    new_audio = AudioFileClip(clean_audio_path)

    final_video = video.set_audio(new_audio)

    final_video.write_videofile(
        str(output_video),
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None
    )


# --------------------------------------------------
# MAIN VIDEO PROCESSING PIPELINE
# --------------------------------------------------

def process_video(video_input, output_video):

    temp_audio = "temp_audio.wav"
    clean_audio_file = "clean_audio.wav"

    print("Extracting audio from video...")
    extract_audio(video_input, temp_audio)

    print("Cleaning audio...")
    clean_audio(temp_audio, clean_audio_file)

    print("Rebuilding video with clean audio...")
    merge_audio_video(video_input, clean_audio_file, output_video)

    print("Processing completed!")
    print(f"Enhanced video saved to: {output_video}")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) == 3:
        video_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
    else:
        video_path = Path(input("Enter input video path: ").strip())
        output_path = Path(input("Enter output video path: ").strip())

    if not video_path.exists():
        print("Input video not found!")
    else:
        process_video(video_path, output_path)