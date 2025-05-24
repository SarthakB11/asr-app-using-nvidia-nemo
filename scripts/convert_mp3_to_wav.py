# scripts/convert_mp3_to_wav.py
import argparse
import os
from pydub import AudioSegment

def convert_to_wav(mp3_file_path, output_wav_path, target_sr=16000, channels=1):
    """
    Converts an MP3 file to a WAV file with specified sample rate and channels.

    Args:
        mp3_file_path (str): Path to the input MP3 file.
        output_wav_path (str): Path to save the output WAV file.
        target_sr (int): Target sample rate for the WAV file.
        channels (int): Number of channels for the WAV file (1 for mono, 2 for stereo).
    """
    try:
        if not os.path.exists(mp3_file_path):
            print(f"Error: Input MP3 file not found at {mp3_file_path}")
            return False

        print(f"Loading MP3 file: {mp3_file_path}")
        audio = AudioSegment.from_mp3(mp3_file_path)
        print(f"Original audio - Channels: {audio.channels}, Frame rate: {audio.frame_rate}")

        print(f"Setting channels to {channels} (mono)")
        audio = audio.set_channels(channels)

        print(f"Setting frame rate to {target_sr} Hz")
        audio = audio.set_frame_rate(target_sr)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_wav_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        print(f"Exporting to WAV: {output_wav_path}")
        audio.export(output_wav_path, format="wav")
        print(f"Successfully converted {mp3_file_path} to {output_wav_path}")
        return True
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        print("Please ensure ffmpeg or libav is installed and in your system's PATH.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP3 to WAV format.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input MP3 file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output WAV file.")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (default: 16000 Hz).")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels (default: 1 for mono).")

    args = parser.parse_args()

    convert_to_wav(args.input, args.output, args.sr, args.channels)
