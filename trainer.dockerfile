# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY setup.py setup.py
COPY requirements.txt requirements.txt

# goto workdir 
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/figures/ reports/figures/

# entrypoint 
ENTRYPOINT ["python3", "-u", "src/models/train_model.py", "train"]
