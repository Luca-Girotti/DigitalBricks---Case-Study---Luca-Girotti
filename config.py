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
SEARCH_TOP_K = 3

# The similarity floor. Chunks scoring below this are thrown away before
# the model ever sees them.
#
# This is the single most important number in the project. Embedding
# similarity always returns *something* - even a question about parking,
# which our knowledge base says nothing about, will return the three
# "least unrelated" chunks with scores around 0.2-0.3. Without a floor,
# the model receives that noise and is tempted to construct a plausible
# sounding policy out of it. That is exactly the hallucination the case
# study warns about.
#
# PROVISIONAL VALUE - not yet measured against this corpus.
#
# 0.35 is a starting point based on how text-embedding-3 models usually
# behave: genuinely unrelated text tends to score 0.1-0.3, genuinely
# relevant text 0.45-0.75, leaving a gap in between for the threshold to
# sit in. That gap has to be confirmed, not assumed.
#
# The number is meaningless on its own. What matters is the separation
# between the two groups for THIS knowledge base. evaluate.py prints the
# top score for every test question, and the threshold belongs in the
# valley between the worst should-pass score and the best should-fail
# score. If those two overlap, no threshold works and the fix is better
# chunking or better documents - not a different number.
MIN_RELEVANCE_SCORE = 0.35


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
