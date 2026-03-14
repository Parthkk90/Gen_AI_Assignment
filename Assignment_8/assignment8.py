from gtts import gTTS


def main() -> None:
    text = "नमस्ते, आप कैसे हैं?."
    tts = gTTS(text=text, lang="hi", slow=False)
    output_file = "output_gtts.mp3"
    tts.save(output_file)
    print(f"Speech generated and saved to {output_file}")


if __name__ == "__main__":
    main()
