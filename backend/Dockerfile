
# HuggingFace Spaces Docker — Teleassist RAG Backend
# Runs ingest scripts at container startup so chroma_store
# is built inside the container — no binary files in git.
 
FROM python:3.11-slim
 
# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
 
WORKDIR /app
 
# Copy and install dependencies first (Docker layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
 
# Copy all app files
COPY --chown=user . /app
 
# Copy data files
COPY --chown=user data/ /app/data/
 
# Run ingest scripts then start the server
# The ; true ensures server starts even if ingest already ran
CMD python ingest_faq.py && \
    python ingest_tickets.py && \
    python ingest_pdf.py && \
    uvicorn api:app --host 0.0.0.0 --port 7860