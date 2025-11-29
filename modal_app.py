import modal
import os
import sys

# Define the base image and install dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "portaudio19-dev",
        "libsndfile1",
        "espeak-ng",
        "git",
    )
    .pip_install(
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "groq",
        "realtimetts[kokoro]==0.5.5",
        "realtimestt==0.3.104",
        "numpy",
        "scipy",
    )
)

# Add your local source directories / files to the image
image = image.add_local_dir("code", remote_path="/root/code")
# image = image.add_local_dir("static", remote_path="/root/static")
if os.path.exists("system_prompt.txt"):
    image = image.add_local_file("system_prompt.txt", remote_path="/root/system_prompt.txt")

app = modal.App("realtime-voice-chat", image=image)

@app.function(
    image=image,
    secrets=[ modal.Secret.from_name("groq-api-key") ],
    gpu="L4",  # Enable GPU for Kokoro TTS acceleration
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    import sys
    import os
    sys.path.insert(0, "/root/code")
    sys.path.append("/root")

    os.environ["TTS_START_ENGINE"] = "kokoro"
    os.environ["LLM_START_PROVIDER"] = "groq"
    os.environ["LLM_START_MODEL"] = "openai/gpt-oss-20b"

    import server
    return server.app
