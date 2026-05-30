"""
main.py
=======
CLI entry point for the Telecom RAG Chatbot.

Demonstrates the full pipeline in a terminal session:
  - Retrieves scored documents with citations
  - Applies the confidence gate (skips LLM if no confident hits)
  - Streams the LLM response token-by-token
  - Prints source citations after each answer

Usage:
  python main.py
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json
from dotenv import load_dotenv
from rag_chain import ask

load_dotenv()

DIVIDER = "─" * 60


def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   Telecom Customer Care Chatbot  (RAG + Groq)   ║")
    print("╚══════════════════════════════════════════════════╝")
    print("Type your question and press Enter.  Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print(f"\nAssistant: ", end="", flush=True)

        collected_sources = []

        # Iterate over SSE events from the generator
        for raw_event in ask(question):
            # Each event looks like: "data: {json}\n\n"
            # Strip the SSE framing to get the JSON payload.
            line = raw_event.strip()
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:"):].strip())

            if payload["type"] == "sources":
                # Store sources; print them after the answer
                collected_sources = payload["data"]

            elif payload["type"] == "token":
                print(payload["data"], end="", flush=True)

            elif payload["type"] == "fallback":
                print(payload["data"], end="", flush=True)

            elif payload["type"] == "error":
                print(f"\n[ERROR] {payload['data']}", flush=True)

            elif payload["type"] == "done":
                break  # end of stream

        # Print cited sources in a tidy block after the answer
        if collected_sources:
            print(f"\n\n{DIVIDER}")
            print("📚 Sources used:")
            for src in collected_sources:
                score_pct = max(0, round((1 - src['score']) * 100, 1))
                print(f"  • {src['citation']}  (relevance: {score_pct}%)")
                print(f"    {src['preview']}")
        print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    main()
