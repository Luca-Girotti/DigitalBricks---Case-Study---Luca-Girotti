"""
The tools the model can call, and the code that runs them.

Two things live here:

1. TOOL_DEFINITIONS - the schemas handed to the model, describing what
   each tool does and what arguments it takes.
2. run_tool() - takes the model's chosen tool and arguments, runs the
   real Python function, and returns the result as text.

The descriptions below are not documentation. They ARE the routing.

The case study asks that the assistant decide for itself whether a
question needs retrieval, the ticket tool, both or neither, with no
keyword matching. That decision is made by the model reading these
descriptions and comparing them against the question. So the useful work
is in writing descriptions that draw a clear line - including saying
what each tool is NOT for, which is what stops the model reaching for
the wrong one on a borderline question.

If routing ever goes wrong, this file is where it is fixed. Not with an
if statement in the agent loop.
"""

import json

import openai

import knowledge
import tickets

# The tool schemas, in the format the OpenAI chat completions API
# expects. The model receives these with every request.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search Northwind's internal policy and supplier documents. "
                "Use this for questions about company policy, procedures, "
                "approval and spending limits, escalation, SLA and priority "
                "definitions, approved suppliers, site access rules, "
                "purchasing and invoicing. Also use it to look up what a "
                "ticket's priority level means. It does NOT know anything "
                "about individual tickets or their status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to search for, in natural language. Search "
                            "is by meaning rather than keyword, so a full "
                            "question works better than a few words. Resolve "
                            "anything the user left implicit first: if they "
                            "asked 'and who approves that?', search for the "
                            "thing 'that' refers to."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": (
                "Look up one maintenance ticket by its reference and return "
                "its status, priority, site, owner and latest note. Use this "
                "whenever the user mentions a specific ticket. It returns "
                "only that ticket's own details - it does NOT explain what "
                "the priority level means or what the SLA is, which are in "
                "the policy documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": (
                            "The ticket reference exactly as the user wrote "
                            "it, for example 'FS-1042'. Pass it through "
                            "unchanged even if it looks wrong - do not tidy "
                            "it up or guess at a correction, because the "
                            "tool reports malformed references back to the "
                            "user so they can fix them themselves."
                        ),
                    }
                },
                "required": ["ticket_id"],
            },
        },
    },
]


def _format_search_results(result: knowledge.SearchResult) -> str:
    """Turn a SearchResult into the text the model will read.

    Each surviving chunk is labelled with its citation and its score, so
    the model has the source to hand when it writes the answer and can
    see which chunk was the strongest match.

    Args:
        result: What knowledge.search() returned.

    Returns:
        The formatted excerpts, or an explicit statement that nothing
        relevant was found. The wording of that empty case matters: a
        blank string invites the model to fall back on its own knowledge,
        whereas a clear sentence tells it there is genuinely nothing to
        work with.
    """
    if not result.matches:
        return (
            "No sufficiently relevant text was found in the knowledge base "
            "for this query. The knowledge base does not appear to cover "
            "this topic. Do not answer from general knowledge."
        )

    blocks = []
    for chunk, score in result.matches:
        blocks.append(f"[Source: {chunk.citation()} | relevance {score:.2f}]\n{chunk.text}")

    return "\n\n---\n\n".join(blocks)


def run_tool(name: str, arguments_json: str) -> tuple[str, dict]:
    """Run one tool the model asked for.

    Args:
        name: The tool name the model chose.
        arguments_json: Its arguments, as the JSON string the API returns.

    Returns:
        A tuple of (text_for_model, metadata).

        text_for_model goes back into the conversation as the tool
        result. metadata is for us, not the model: it carries the
        retrieval score and the citations so that evaluate.py can check
        grounding and tune the relevance threshold. Without it we could
        only inspect the final prose, which tells us what the assistant
        said but not what it was working from.

    Never raises for anything the model or Azure can cause. Unparseable
    arguments, an unknown tool name, a missing argument, or an Azure error
    during search all come back as text the model can read and explain to
    the user, because an exception here would end the turn over something
    recoverable. A genuine bug in our own code is left to fail loudly.
    """
    # The model produces these arguments as a JSON string. It is almost
    # always valid, but "almost always" is not a guarantee we can build
    # on when the result is a crash mid-conversation.
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        return (
            f"Could not read the arguments for {name}. Please try again "
            "with a simpler request.",
            {},
        )

    if name == "search_documents":
        query = arguments.get("query", "")
        if not query:
            return ("No search query was provided.", {})

        try:
            result = knowledge.search(query)
        except openai.APIError as error:
            # Azure failed: a timeout, a rate limit, or - as happened once
            # during testing - the embedding deployment briefly returning
            # 404. Before this was caught, one failed search ended the
            # whole turn, taking a perfectly good ticket answer down with
            # it on a mixed question. Now the model is told search is
            # unavailable and can still answer everything else.
            #
            # Only Azure's own errors are caught. A bug in our code is not
            # an APIError and should still fail loudly.
            return (
                "The document search is temporarily unavailable "
                f"({type(error).__name__}). Tell the user you could not "
                "check the policy documents right now and suggest trying "
                "again shortly. Do not answer the policy part from general "
                "knowledge.",
                {"search_error": type(error).__name__},
            )

        metadata = {
            "top_score": result.top_score,
            "sources": [chunk.citation() for chunk, _ in result.matches],
        }
        return _format_search_results(result), metadata

    if name == "get_ticket_status":
        ticket = tickets.get_ticket_status(arguments.get("ticket_id", ""))
        # json.dumps because tool results must be strings, and the model
        # reads JSON perfectly well.
        return json.dumps(ticket), {"ticket_found": ticket["found"]}

    # Only reachable if TOOL_DEFINITIONS and this function disagree -
    # which is exactly the sort of thing worth catching out loud.
    return (f"Unknown tool: {name}", {})
