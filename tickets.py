"""
A mock of the line-of-business ticket system.

In the real Northwind setup this would be an HTTP call to their service
management platform. Here it is a dictionary, which is all the case study
asks for - the point being assessed is the tool-calling pattern, not the
backend.

The important design decision in this file is that get_ticket_status()
*never raises an exception*. It always returns a dictionary describing
what happened. That is because its output is fed straight back to the
language model as a tool result, and a model can reason about
{"found": false, "reason": "..."} far more usefully than it can about a
crash. Every failure mode becomes a piece of information the assistant
can explain to the user.
"""

import re

# The shape a valid Northwind ticket reference takes: the letters FS, a
# hyphen, then exactly four digits. Compiled once at import rather than
# on every call.
#
# re.IGNORECASE so that "fs-1042" typed in lower case is accepted - users
# should not be punished for not holding shift.
TICKET_ID_PATTERN = re.compile(r"^FS-\d{4}$", re.IGNORECASE)


# The fake database. Eight tickets, chosen to cover every priority level
# and every status, so that the evaluation questions have something
# meaningful to test against.
TICKETS = {
    "FS-1042": {
        "ticket_id": "FS-1042",
        "site": "Antwerp",
        "summary": "No cooling to the second floor open-plan area",
        "priority": "P2",
        "status": "In progress",
        "assigned_to": "Sanne de Vries",
        "opened": "2026-09-18",
        "last_update": "2026-09-20",
        "notes": "Filters replaced, fault persists. Thermex attending Monday.",
    },
    "FS-1043": {
        "ticket_id": "FS-1043",
        "site": "Amsterdam",
        "summary": "Main entrance access control door not releasing",
        "priority": "P2",
        "status": "Resolved",
        "assigned_to": "Tom Bakker",
        "opened": "2026-09-15",
        "last_update": "2026-09-16",
        "notes": "Strike plate realigned. Monitored for 24 hours, no recurrence.",
    },
    "FS-1051": {
        "ticket_id": "FS-1051",
        "site": "Rotterdam",
        "summary": "Total power loss to the east wing",
        "priority": "P1",
        "status": "Resolved",
        "assigned_to": "Tom Bakker",
        "opened": "2026-09-11",
        "last_update": "2026-09-11",
        "notes": "Main breaker tripped by a faulty UPS. UPS isolated pending replacement.",
    },
    "FS-1077": {
        "ticket_id": "FS-1077",
        "site": "Brussels",
        "summary": "Lift 2 out of service",
        "priority": "P2",
        "status": "Waiting on supplier",
        "assigned_to": "Marie Dubois",
        "opened": "2026-09-17",
        "last_update": "2026-09-19",
        "notes": "Otis attended, door operator part on order. ETA 24 September.",
    },
    "FS-1080": {
        "ticket_id": "FS-1080",
        "site": "Antwerp",
        "summary": "Three failed light fittings in the ground floor corridor",
        "priority": "P3",
        "status": "Open",
        "assigned_to": "Unassigned",
        "opened": "2026-09-19",
        "last_update": "2026-09-19",
        "notes": "Lamps in stock. To be picked up on the next planned visit.",
    },
    "FS-1091": {
        "ticket_id": "FS-1091",
        "site": "Amsterdam",
        "summary": "Relocate eight desks from floor 3 to floor 5",
        "priority": "P4",
        "status": "Scheduled",
        "assigned_to": "Tom Bakker",
        "opened": "2026-09-08",
        "last_update": "2026-09-12",
        "notes": "Booked for the weekend of 26 September, out of hours, client approved.",
    },
    "FS-1102": {
        "ticket_id": "FS-1102",
        "site": "Rotterdam",
        "summary": "Water ingress in the basement plant room",
        "priority": "P1",
        "status": "In progress",
        "assigned_to": "Sanne de Vries",
        "opened": "2026-09-20",
        "last_update": "2026-09-21",
        "notes": "Leak isolated. Drying equipment in place. Cause under investigation.",
    },
    "FS-1115": {
        "ticket_id": "FS-1115",
        "site": "Brussels",
        "summary": "Client complaint regarding cleaning standards",
        "priority": "P3",
        "status": "Open",
        "assigned_to": "Marie Dubois",
        "opened": "2026-09-21",
        "last_update": "2026-09-21",
        "notes": "Raised by the client's office manager. Site walk arranged for Tuesday.",
    },
}


def get_ticket_status(ticket_id: str) -> dict:
    """Look up a single ticket by its reference.

    This is the function the language model calls as a tool. The model
    supplies whatever it extracted from the user's message, which means
    the argument can be anything at all - the wrong format, an empty
    string, or a ticket that simply does not exist. All three are handled
    here rather than being allowed to become an exception.

    Args:
        ticket_id: The ticket reference as the model extracted it, for
            example "FS-1042". Case is not significant and surrounding
            whitespace is ignored.

    Returns:
        A dictionary that always contains a "found" key:

        On success, found is True and the ticket's fields are included.

        On failure, found is False and "reason" explains which of the
        three failure modes occurred, in wording the model can pass on to
        the user. For an unknown but well-formed ID we also return
        "known_ticket_ids" so the assistant can suggest alternatives
        instead of leaving the user stuck.

    Examples:
        >>> get_ticket_status("FS-1042")["status"]
        'In progress'
        >>> get_ticket_status("banana")["found"]
        False
    """
    # Guard against the model passing None or a non-string. Defensive,
    # but tool arguments come from a model and are not guaranteed.
    if not isinstance(ticket_id, str) or not ticket_id.strip():
        return {
            "found": False,
            "reason": "No ticket reference was provided.",
        }

    cleaned = ticket_id.strip().upper()

    # Failure mode 1: the reference is not shaped like a ticket at all.
    # Worth separating from "not found" because the advice differs - here
    # the user has probably mistyped, rather than asked about a ticket
    # that does not exist.
    if not TICKET_ID_PATTERN.match(cleaned):
        return {
            "found": False,
            "reason": (
                f"'{ticket_id}' is not a valid ticket reference. "
                "Northwind ticket references look like FS-1042: "
                "the letters FS, a hyphen, then four digits."
            ),
        }

    # Failure mode 2: correctly formatted, but no such ticket.
    if cleaned not in TICKETS:
        return {
            "found": False,
            "reason": f"No ticket found with reference {cleaned}.",
            "known_ticket_ids": sorted(TICKETS.keys()),
        }

    # Success. Copy the record so a caller cannot accidentally mutate the
    # module-level dictionary - cheap insurance in a long-running chat.
    ticket = TICKETS[cleaned].copy()
    ticket["found"] = True
    return ticket
