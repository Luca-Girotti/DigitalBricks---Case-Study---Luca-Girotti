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
# 0.35 was chosen by running evaluate.py and looking at the actual scores
# for questions that should succeed versus questions that should fail.
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
