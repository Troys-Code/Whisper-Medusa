# main.py

import argparse
from TTS import SpeechToText

def main():
    parser = argparse.ArgumentParser(description="Transcribe speech to text using Whisper Medusa.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    args = parser.parse_args()

    stt = SpeechToText()
    transcription = stt.transcribe(args.audio_file)
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()
