from src.examples.metrics import all_metrics
from src.llms.managers.ollama_manager import ollama_manager
from ollama import ResponseError

if __name__ == "__main__":

    ollama_manager.start_ollama()
    while True:
        try:
            all_metrics()
        except ResponseError as e:
            print(f"Error: {e}")
            ollama_manager.quick_restart()
