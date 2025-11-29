import modal
import os
import sys

# Use PyTorch official image with CUDA 12.4 + cuDNN 9 pre-installed
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install(
        "build-essential",
        "python3-dev",
        "ffmpeg",
        "portaudio19-dev",
        "libsndfile1",
        "espeak-ng",
        "git",
        "clang",
    )
    # Suppress ALSA errors
    .run_commands(
        'echo "pcm.!default { type plug slave.pcm null }" > /etc/asound.conf'
    )
    .pip_install(
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "groq",
        "numpy",
        "scipy",
    )
    # PyTorch is already installed in the base image with correct CUDA/cuDNN
    # Just install TTS libraries
    .pip_install(
        "realtimetts[kokoro]==0.5.5",
        "realtimestt==0.3.104",
    )
)

image = image.add_local_dir("code", remote_path="/root/code")
if os.path.exists("system_prompt.txt"):
    image = image.add_local_file("system_prompt.txt", remote_path="/root/system_prompt.txt")

app = modal.App("realtime-voice-chat", image=image)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("groq-api-key")],
    gpu="L4",
    scaledown_window=60,
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