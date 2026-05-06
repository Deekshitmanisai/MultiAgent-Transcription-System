import tempfile

import ffmpeg

from config import TEMP_DIR
from runtime_utils import ensure_ffmpeg_in_path


def extract_audio(video_path, output_audio=None):
    """
    Extract high-quality audio from video and format it for speech recognition.
    
    Parameters:
        video_path (str): Path to input video file
        output_audio (str): Path to output audio file

    Returns:
        str: Path to extracted audio file
    """

    TEMP_DIR.mkdir(exist_ok=True)
    ffmpeg_executable = ensure_ffmpeg_in_path()

    if output_audio is None:
        with tempfile.NamedTemporaryFile(
            prefix="audio_",
            suffix=".wav",
            dir=TEMP_DIR,
            delete=False,
        ) as temp_file:
            output_audio = temp_file.name

    try:
        (
            ffmpeg
            .input(video_path)
            .output(
                output_audio,
                acodec='pcm_s16le',
                ac=2,
                ar='16000'           
            )
            .run(cmd=ffmpeg_executable or "ffmpeg", overwrite_output=True)
        )

        print(f"Audio extracted and formatted for ASR successfully: {output_audio}")
        return output_audio

    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure it is installed and added to your PATH.")
        return None
    except Exception as e:
        print("Error extracting audio:", e)
        return None


if __name__ == "__main__":

    video_file = "sample_video.mp4"

    audio = extract_audio(video_file)

    print("Generated audio file:", audio)
