"""
The agent loop.

This is the whole system in one short function. Everything else in the
project exists to support it.

The loop:

    1. Send the conversation to the model, along with the tool schemas.
    2. If the model replied with text, that is the answer. Stop.
    3. If it asked to call tools, run them, append the results to the
       conversation, and go back to step 1.

That is all the routing there is. We never inspect the question, never
match keywords, never decide anything on the model's behalf. The model
sees two tool descriptions and picks none, one, or both. A question that
needs a ticket and a policy comes back round the loop twice - once to
look up the ticket, then again to look up what its priority means.

Why hand-rolled rather than Semantic Kernel or the Agent Service: the
loop is about twenty lines. A framework would hide those twenty lines
behind an abstraction and add a dependency, in exchange for features
(planners, memory stores, multi-agent handoff) that this assistant does
not need. Keeping it explicit means every decision the system makes is
visible in one file.
"""

from dataclasses import dataclass, field

import config
import tools
from prompts import SYSTEM_PROMPT


@dataclass
class AgentReply:
    """One answer, plus what it took to produce it.

    The extra fields are not shown to the user. They exist so that
    evaluate.py can check HOW an answer was reached rather than just
    reading the prose - whether the right tools were chosen, whether
    anything was actually retrieved, and how strong the match was.

    Attributes:
        answer: The assistant's reply, as shown to the user.
        tools_called: Names of every tool called, in order. This is the
            routing decision, recorded.
        sources: Citations of the chunks that cleared the relevance
            threshold. Empty means the answer had no document backing.
        top_score: The best retrieval score seen. Reported even when
            nothing cleared the threshold, so a refusal can be checked
            for being a near miss or a clear miss.
    """

    answer: str
    tools_called: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    top_score: float = 0.0


class Assistant:
    """A single conversation with the Northwind service desk assistant.

    Holds the message history, which is what gives the assistant its
    memory. Follow-ups like "and who approves that?" work because every
    previous turn is still in the list we send.

    One instance is one conversation. evaluate.py builds a fresh
    Assistant per test case so that cases cannot contaminate each other.
    """

    def __init__(self) -> None:
        """Start a conversation with the system prompt in place."""
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(self, question: str) -> AgentReply:
        """Ask a question and get the assistant's reply.

        Args:
            question: The user's message.

        Returns:
            An AgentReply with the answer and the details of how it was
            reached.
        """
        self.messages.append({"role": "user", "content": question})

        client = config.get_client()
        tools_called: list[str] = []
        sources: list[str] = []
        top_score = 0.0

        # Bounded rather than while True. A model that keeps calling
        # tools without ever answering would otherwise loop forever,
        # running up an Azure bill with nobody watching.
        for _ in range(config.MAX_TOOL_ITERATIONS):
            response = client.chat.completions.create(
                model=config.CHAT_DEPLOYMENT,
                messages=self.messages,
                tools=tools.TOOL_DEFINITIONS,
                temperature=config.TEMPERATURE,
            )

            message = response.choices[0].message

            # The model's own turn goes into the history either way. If
            # it requested tools, the API requires that this message is
            # present before the tool results that answer it.
            self.messages.append(message)

            # No tool calls means the model is done and this is the
            # answer.
            if not message.tool_calls:
                return AgentReply(
                    answer=message.content or "",
                    tools_called=tools_called,
                    sources=sources,
                    top_score=top_score,
                )

            # Otherwise run everything it asked for. The model can
            # request several tools at once - the mixed ticket-and-policy
            # question often arrives as two calls in a single turn.
            for call in message.tool_calls:
                tools_called.append(call.function.name)

                result_text, metadata = tools.run_tool(
                    call.function.name, call.function.arguments
                )

                sources.extend(metadata.get("sources", []))
                top_score = max(top_score, metadata.get("top_score", 0.0))

                # tool_call_id ties the result back to the specific
                # request, which is how the model knows which answer
                # belongs to which call.
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result_text,
                    }
                )

        # Fell out of the loop still calling tools. Rare, but it must not
        # surface as a crash or an empty reply.
        return AgentReply(
            answer=(
                "I wasn't able to settle on an answer for that. Could you "
                "try rephrasing it, or splitting it into separate questions?"
            ),
            tools_called=tools_called,
            sources=sources,
            top_score=top_score,
        )
