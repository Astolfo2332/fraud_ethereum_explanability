import os
import subprocess
import time
from langchain_ollama import ChatOllama
import signal

from ollama import ResponseError

env = os.environ.copy()
env["OLLAMA_MODELS"] = "/mnt/1_tera_linux/models"

class OllamaManager:
    def __init__(self, model:str=None):
        self.model = None

    def set_model(self, model:str):
        self.model = model

    def get_active_models(self) -> list[str]:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        lines = result.stdout.strip()
        if len(lines) > 1:
            return lines[1].split()[0]
        return []

    def stop_model(self):
        if self.model is not None:
            os.system(f"ollama stop {self.model}")
            self.model = None

    def kill_ollama(self):
        try:
            pid = subprocess.check_output(
                "pgrep -f 'ollama serve'",
                shell=True
            ).decode().strip()
            os.kill(int(pid), signal.SIGTERM)
        except subprocess.CalledProcessError:
            pass

    def start_ollama(self):
                subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp,
            env=env
        )

    def quick_restart(self):
        print("Restarting Ollama... waiting 60 seconds")
        self.kill_ollama()
        time.sleep(50)
        self.start_ollama()
        time.sleep(10)

ollama_manager = OllamaManager()

if __name__ == "__main__":
    def ollama_switcher():
        model = ChatOllama(model="gemma3:12b", temperature=0.0)
        r = model.invoke("Hello, how are you?")
        model = ChatOllama(model="gpt-oss:20b", temperature=0.0)
        r = model.invoke("Hello, how are you?")

    try:
        while True:
            ollama_switcher()
    except ResponseError:
        print("Restarting Ollama...")
        ollama_manager.quick_restart()
        ollama_switcher()

