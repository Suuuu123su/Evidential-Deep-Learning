import importlib.util
import sys


def _spec(name: str) -> str:
    spec = importlib.util.find_spec(name)
    return "FOUND" if spec is not None else "MISSING"


def main() -> None:
    print(f"python: {sys.executable}")
    print(f"version: {sys.version}")
    for name in ["numpy", "torch", "torchvision", "tqdm"]:
        print(f"{name}: {_spec(name)}")

    if importlib.util.find_spec("torch") is not None:
        import torch

        print(f"torch.__version__: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()

