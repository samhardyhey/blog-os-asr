import argparse

from asr import transcribe_mono_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="transcribe_audio_os",
        description="Transcribe audio using open-source utilities and models",
    )
    parser.add_argument("input_path", type=str, help="Input audio file")
    parser.add_argument(
        "output_path", type=str, help="Output transcript path, saves as a csv"
    )
    args = parser.parse_args()

    transcription = transcribe_mono_audio(input_file=args.input_path)
    transcription.to_csv(args.output_path, index=False)
