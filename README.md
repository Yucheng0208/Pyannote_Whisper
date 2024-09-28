# Speaker Diarization and Transcription Pipeline

This project is a Python-based tool that integrates Whisper for transcription and Pyannote for speaker diarization. It processes `.wav` files, transcribes them using Whisper, and matches the transcriptions to speakers identified by Pyannote.

## Features
- **Automatic Speech Recognition** using OpenAI's Whisper model with automatic language detection.
- **Speaker Diarization** using Pyannote to detect different speakers in an audio file.
- **Audio Format Conversion**: Uses `ffmpeg` to convert audio files to the required format.
- **Integrated Transcription and Speaker Matching**: Combines Whisper transcription segments with Pyannote speaker diarization.

## Prerequisites
- Python 3.8+
- [Whisper](https://github.com/openai/whisper)
- [Pyannote](https://github.com/pyannote/pyannote-audio)
- [ffmpeg](https://ffmpeg.org/download.html) for audio format conversion
- PyTorch and Torchaudio for handling audio

You can install the necessary Python packages via `pip`:

```pip install torch torchaudio whisper pyannote.audio```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

