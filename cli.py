"""
The front end: a terminal chat.

Run it with:

    python cli.py

The case study says front-end polish carries very little weight and that
a clean CLI is a perfectly good choice, so this is deliberately thin. It
reads a line, passes it to the Assistant, prints the reply, repeats.

The one extra is that it prints which tools were called for each answer.
That is a debugging aid rather than decoration: routing is the part of
the system most worth watching, and seeing it live is more useful during
a demo than inferring it from the prose.
"""

import config
from agent import Assistant

BANNER = """
Northwind Facilities - Service Desk Assistant
---------------------------------------------
Ask about policy, suppliers, or a ticket (for example FS-1042).
Type 'quit' to exit.
"""


def main() -> None:
    """Run the chat loop until the user quits."""
    # Check configuration before printing a welcome, so a missing key
    # gives one clear sentence rather than a banner followed by a stack
    # trace on the first question.
    try:
        config.validate_config()
    except RuntimeError as error:
        print(f"\nConfiguration problem:\n\n{error}\n")
        return

    print(BANNER)

    # One Assistant for the whole session, so it remembers the
    # conversation and follow-up questions work.
    assistant = Assistant()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-C or Ctrl-D should look like a normal exit rather than
            # a traceback.
            print("\nGoodbye.")
            return

        if not question:
            continue

        if question.lower() in {"quit", "exit", "bye"}:
            print("Goodbye.")
            return

        try:
            reply = assistant.ask(question)
        except Exception as error:
            # Azure can time out, rate-limit, or refuse. None of those
            # should end the session - the user may simply want to ask
            # again in a moment.
            print(f"\n[error] {type(error).__name__}: {error}\n")
            continue

        print(f"\nAssistant: {reply.answer}\n")

        if reply.tools_called:
            used = ", ".join(reply.tools_called)
            print(f"           [tools: {used} | best match {reply.top_score:.2f}]\n")
        else:
            print("           [tools: none]\n")


if __name__ == "__main__":
    main()
