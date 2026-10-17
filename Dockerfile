FROM python:3.11.1-slim

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model into the image so cold starts are fast
RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')"

COPY handler.py .

CMD ["python", "-u", "/handler.py"]
