FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip git python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
