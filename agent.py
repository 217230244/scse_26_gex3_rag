"""Run the RAG assistant on the graded exercise question."""

import argparse

from rag import RAGAssistant


DEFAULT_QUESTION = "What exact phone number should I call for the University IT Service Desk?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    print(RAGAssistant(model_path=args.model_path).answer(args.question))


if __name__ == "__main__":
    main()
