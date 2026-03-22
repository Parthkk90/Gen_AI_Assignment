import os
import time
from google import genai
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file.")

    client = genai.Client(api_key=api_key)

    prompt = (
        "A close up of two people staring at a cryptic drawing on a wall, torchlight flickering. "
        "A man murmurs, 'This must be it. That's the secret code.' The woman looks at him and "
        "whispering excitedly, 'What did you find?'"
    )

    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
    )

    while not operation.done:
        print("Waiting for video generation to complete...")
        time.sleep(10)
        operation = client.operations.get(operation)

    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    generated_video.video.save("dialogue_example.mp4")
    print("Generated video saved to dialogue_example.mp4")


if __name__ == "__main__":
    main()
