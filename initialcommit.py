#!/usr/bin/env python3
"""
bootstrap.py - Initialize the heart-disease-gpu repository.
"""

import subprocess
import sys
from pathlib import Path

# Configuration: Define your project layout here
PROJECT_DIRS = [
    "src/cuda_logreg",
    "notebooks",
    "tests",
    "data",
    ".github/workflows",
]

PROJECT_FILES = {
    "requirements.txt": (
        "pandas==2.2.3\n"
        "scikit-learn==1.5.2\n"
        "matplotlib==3.9.2\n"
        "seaborn==0.13.2\n"
        "jupyterlab==4.2.5\n"
        "pytest==8.3.3\n"
        "pybind11==2.13.6\n"
        "numba==0.60.0\n"
        "plotly==5.24.1\n"
        "torch\n"
    ),
    ".gitignore": (
        "# Python\n"
        "__pycache__/\n*.py[cod]\n*.so\n"
        "env/\nvenv/\n"
        "\n# Jupyter\n"
        ".ipynb_checkpoints/\n"
        "\n# CUDA\n"
        "*.cu.o\n*.ptx\n*.cubin\n"
        "\n# OS\n"
        ".DS_Store\n"
    ),
    "Dockerfile": (
        "FROM nvidia/cuda:12.2.0-base-ubuntu22.04\n\n"
        "RUN apt-get update && apt-get install -y \\\n"
        "    python3-pip git python3-dev build-essential \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n\n"
        "WORKDIR /app\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n\n"
        "EXPOSE 8888\n"
        'CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]\n'
    ),
    "README.md": (
        "# GPU-Accelerated Heart Disease Prediction\n\n"
        "Accelerating Logistic Regression kernels using CUDA.\n\n"
        "## Setup\n"
        "1. Build Docker: docker build -t heart-gpu .\n"
        "2. Run: docker run -it --gpus all -p 8888:8888 heart-gpu\n"
    ),
    "src/cpu_baseline.py": "# CPU Implementation\n",
    "src/cuda_logreg/kernel.cu": "// CUDA kernels\n",
    "src/cuda_logreg/host.cu": "// Host-side C++ code\n",
    "tests/test_pipeline.py": "def test_placeholder():\n    assert True\n",
}

def run_cmd(args, check=True):
    """Executes a system command without shell=True for safety."""
    print(f"  Executing: {' '.join(args)}")
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=check)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("ERROR: Command failed")
        print(f"STDERR: {e.stderr}")
        if check:
            sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Command '{args[0]}' not found.")
        sys.exit(1)

def setup_workspace():
    """Creates the directory structure and files."""
    print("--- Creating Structure ---")
    for d in PROJECT_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created directory: {d}")

    for file_path, content in PROJECT_FILES.items():
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="ascii") # Explicitly use ASCII
        print(f"  Created file: {file_path}")

def init_git():
    """Initializes git and performs the first commit."""
    print("\n--- Initializing Git ---")
    
    if Path(".git").exists():
        print("  Git repository already exists. Skipping init.")
        return

    run_cmd(["git", "init"])
    run_cmd(["git", "checkout", "-b", "feature/initial-setup"], check=False)
    run_cmd(["git", "add", "."])
    
    commit_msg = (
        "feat: bootstrap heart-disease-gpu project\n\n"
        "- Modern MLOps structure\n"
        "- Dockerized CUDA environment\n"
        "- Initial CUDA/CPU stubs"
    )
    run_cmd(["git", "commit", "-m", commit_msg])

def main():
    print("Bootstrapping heart-disease-gpu\n")
    setup_workspace()
    init_git()
    
    print("\nProject successfully initialized!")
    print("\nNext steps:")
    print("  1. git remote add origin https://github.com/manojhunasimarad/heart-disease-gpu.git")
    print("  2. git push -u origin feature/initial-setup")

if __name__ == "__main__":
    main()