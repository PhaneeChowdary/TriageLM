"""Derive priority, requires_human, suggested_queue from gold intent/category.

An authored routing policy, not ground-truth labels. Pure functions of the gold fields.
"""

from __future__ import annotations

QUEUE_BY_CATEGORY = {
    "ACCOUNT": "account",
    "SUBSCRIPTION": "account",
    "ORDER": "orders",
    "CANCEL": "orders",
    "SHIPPING": "logistics",
    "DELIVERY": "logistics",
    "INVOICE": "billing",
    "PAYMENT": "billing",
    "REFUND": "billing",
    "CONTACT": "support",
    "FEEDBACK": "feedback",
}

ESCALATE_INTENTS = frozenset({"complaint", "contact_human_agent", "payment_issue"})

HIGH_INTENTS = frozenset({
    "payment_issue", "complaint", "contact_human_agent", "get_refund",
    "track_refund", "cancel_order", "check_cancellation_fee",
})
MEDIUM_INTENTS = frozenset({
    "change_order", "change_shipping_address", "set_up_shipping_address",
    "track_order", "delivery_period", "recover_password", "delete_account",
    "registration_problems", "check_refund_policy", "check_invoice",
    "get_invoice", "check_payment_methods",
})


def priority(intent: str) -> str:
    if intent in HIGH_INTENTS:
        return "high"
    if intent in MEDIUM_INTENTS:
        return "medium"
    return "low"


def requires_human(intent: str) -> bool:
    return intent in ESCALATE_INTENTS


def suggested_queue(category: str) -> str:
    return QUEUE_BY_CATEGORY[category]


def derive(intent: str, category: str) -> dict:
    return {
        "priority": priority(intent),
        "requires_human": requires_human(intent),
        "suggested_queue": suggested_queue(category),
    }
