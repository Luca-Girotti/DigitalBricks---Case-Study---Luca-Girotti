"""
All configuration for the Northwind Support Assistant lives here.

Two kinds of thing are in this file:

1. Azure connection details, read from environment variables so that no
   key or endpoint is ever written into the source code.
2. The handful of numbers that control how the assistant behaves. These
   are the "knobs" - if you want to change how many search results come
   back, or how strict the assistant is about refusing to answer, this is
   the only file you need to open.

Keeping these in one place is deliberate: it means a reviewer can see
every tunable decision at a glance instead of hunting through the logic.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read the .env file (if present) and put its contents into the process
# environment. Called once, at import time, so every other module can
# simply read from this file.
load_dotenv()


# --------------------------------------------------------------------------
# Azure OpenAI connection
# --------------------------------------------------------------------------
# os.getenv returns None when a variable is missing. We do not crash here;
# validate_config() below turns a missing value into a friendly message.

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# In Azure you give each deployed model your own name. These are those
# names, not the underlying model names.
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)


# --------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------

# Where the Markdown documents live. Path(__file__).parent is the folder
# containing this file, so the app works no matter where you run it from.
KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"


# --------------------------------------------------------------------------
# Retrieval behaviour - the knobs
# --------------------------------------------------------------------------

# How many document chunks to hand back to the model for a single search.
# Too few and the answer misses context; too many and the model gets
# distracted by irrelevant text and is more likely to invent something.
#
# Was 3. Raised to 5 after the evaluation found a false refusal: asked
# "how quickly do we respond to a P1?", the model searched "the SLA for
# a P1 ticket" and the P1 chunk came 4th - scoring 0.391 against 0.395
# for P2. Embeddings are poor at telling short codes like P1 and P2
# apart, so the four priority chunks score within a few hundredths of
# each other and their order is close to random. With only 3 slots, the
# right one could be cut off and the assistant would then (correctly,
# given what it was shown) say it could not find the answer.
#
# The rule this follows: top-k must be larger than the biggest family of
# near-identical chunks. There are four priority levels, so 5. The
# embedding does rough recall; the model, which reads "P1" perfectly
# well, picks the right chunk from the ones returned. The cost is two
# extra short chunks per search - roughly 250 tokens.
SEARCH_TOP_K = 5

# The similarity floor. Chunks scoring below this are thrown away before
# the model ever sees them.
#
# Similarity search always returns a ranking, even when nothing in the
# corpus is relevant, so without a floor the model can be handed
# unrelated text and tempted to build an answer out of it.
#
# MEASURED, not guessed. Run evaluate.py to reproduce. On this corpus:
#
#     answerable questions        0.40 - 0.80
#     near-domain, unanswerable   0.36 - 0.63   (parking, Berlin, holiday)
#     off-topic                   0.07 - 0.13   (weather, football)
#
# The first two groups overlap: "which electrical supplier do we use in
# Berlin?" scores 0.63 because the supplier directory genuinely is about
# electrical suppliers, which is higher than several real questions. No
# threshold can separate overlapping groups, so near-domain questions are
# refused by the system prompt, not by this number.
#
# That leaves this threshold one job: keep off-topic noise away from the
# model without ever blocking a genuine answer. 0.27 is the midpoint of
# the gap between the highest off-topic score (0.13) and the lowest
# answerable one (0.40), leaving a margin of about 0.13 on each side.
#
# It is a backstop. In testing the agent never searched for off-topic
# questions at all - it declined them without calling a tool - so the
# threshold did not fire once. It is kept because the prompt and the
# routing can fail, and a different model may be more eager to search.
MIN_RELEVANCE_SCORE = 0.27


# --------------------------------------------------------------------------
# Agent behaviour
# --------------------------------------------------------------------------

# A safety stop. The agent loop lets the model call tools, see the
# results, and then call more tools if it needs to. This caps how many
# times round that loop it can go, so a confused model cannot spin
# forever running up an Azure bill.
#
# In practice two passes is normal (one to call tools, one to answer).
# Five leaves room for the model to follow up on a first result.
MAX_TOOL_ITERATIONS = 5

# Lower temperature means less creative, more literal output. For a
# grounded assistant that is what we want - we are explicitly asking it
# not to be imaginative.
TEMPERATURE = 0.1


# A single shared Azure client, created the first time it is asked for
# and then reused. Two modules need one - knowledge.py for embeddings and
# agent.py for chat - and building it in one place keeps the connection
# details in the same file as the settings they come from.
_client = None


def get_client():
    """Return the shared AzureOpenAI client, creating it on first use.

    The import sits inside the function rather than at the top of the
    file so that importing config does not require the openai package or
    a valid key. That matters because knowledge.py and eval_questions.py
    can both be used - for chunking and for reading the test cases -
    with no Azure connection at all.

    Returns:
        An AzureOpenAI client configured from the environment.

    Raises:
        RuntimeError: via validate_config(), if the endpoint or key is
            missing.
    """
    global _client

    if _client is None:
        validate_config()
        from openai import AzureOpenAI

        _client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    return _client


def validate_config() -> None:
    """Check that the required Azure settings are present.

    Raises:
        RuntimeError: if the endpoint or API key is missing, with a
            message telling the user exactly how to fix it.

    Called at startup by the CLI so the failure is a clear sentence
    rather than a confusing error from deep inside the OpenAI SDK.
    """
    missing = []

    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")

    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + "\n\nCopy .env.example to .env and fill in the values:"
            + "\n    cp .env.example .env"
        )
