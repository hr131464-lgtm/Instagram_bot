from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from app.config import (
    META_VERIFY_TOKEN,
    META_INSTAGRAM_ACCOUNT_ID,
)
from app.memory import ConversationMemory
from app.rag import ask
from app.instagram.sender import (
    send_instagram_message,
    reply_to_comment,
)

app = FastAPI(
    title="AI Learning Hub Instagram RAG Bot"
)

conversation_memories = {}

# Comment IDs we've already replied to. Meta can redeliver the same
# webhook event, and our own reply to a comment shows up as a new
# comment event, so we need this to avoid an infinite reply loop.
processed_comment_ids = set()


@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params

    mode = params.get("hub.mode")
    verify_token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if (
        mode == "subscribe"
        and verify_token == META_VERIFY_TOKEN
    ):
        return PlainTextResponse(challenge)

    return PlainTextResponse(
        "Verification failed",
        status_code=403
    )


@app.post("/webhook")
async def receive_webhook(request: Request):
    payload = await request.json()

    print("\nIncoming Instagram webhook:")
    print(payload)

    for entry in payload.get("entry", []):

        # DM events (Instagram Messaging) arrive in a 'messaging'
        # array on the entry, Messenger-platform style - NOT inside
        # 'changes'. Each item already has 'sender' and 'message'
        # keys, same shape handle_message_change() expects.
        for messaging_event in entry.get("messaging", []):
            handle_message_change(messaging_event)

        # Comments / live_comments / mentions etc. arrive in the
        # 'changes' array instead, each with a 'field' + 'value'.
        for change in entry.get("changes", []):

            field = change.get("field")

            if field in ("comments", "live_comments"):
                handle_comment_change(change.get("value", {}))

    return {
        "status": "ok"
    }


def handle_message_change(value: dict):
    """
    Handle an incoming Instagram DM and reply automatically using
    the RAG pipeline.
    """

    sender = value.get("sender", {})
    message = value.get("message", {})

    sender_id = sender.get("id")
    message_text = message.get("text")

    if not sender_id or not message_text:
        return

    # Instagram bundles "echoes" of our own sent messages into the
    # same 'messaging' array. Ignore anything we sent ourselves.
    if message.get("is_echo") or sender_id == META_INSTAGRAM_ACCOUNT_ID:
        return

    print(f"\nSender ID: {sender_id}")
    print(f"Message: {message_text}")

    if sender_id not in conversation_memories:
        conversation_memories[sender_id] = ConversationMemory()

    memory = conversation_memories[sender_id]

    try:
        result = ask(
            message_text,
            memory
        )

        answer = result["answer"]

        print(f"RAG answer: {answer}")

        send_result = send_instagram_message(
            sender_id,
            answer
        )

        print(f"Instagram response: {send_result}")

    except Exception as error:
        print(f"Error processing message: {error}")


def handle_comment_change(value: dict):
    """
    Handle a new comment on one of our posts/reels/live streams
    ('comments' / 'live_comments' webhook field) and reply
    automatically as a public comment reply, using the RAG
    pipeline.
    """

    comment_id = value.get("id")
    comment_text = value.get("text")
    from_user = value.get("from", {})
    commenter_id = from_user.get("id")

    if not comment_id or not comment_text:
        return

    # Skip comments made by our own account (this includes the
    # reply we just posted, which would otherwise loop forever).
    if commenter_id and commenter_id == META_INSTAGRAM_ACCOUNT_ID:
        return

    # Meta may redeliver the same webhook event on retries.
    if comment_id in processed_comment_ids:
        return

    processed_comment_ids.add(comment_id)

    print(f"\nComment ID: {comment_id}")
    print(f"Commenter ID: {commenter_id}")
    print(f"Comment: {comment_text}")

    # Comments are one-off, so no multi-turn memory per commenter.
    memory = ConversationMemory()

    try:
        result = ask(
            comment_text,
            memory
        )

        answer = result["answer"]

        print(f"RAG answer: {answer}")

        send_result = reply_to_comment(
            comment_id,
            answer
        )

        print(f"Instagram response: {send_result}")

    except Exception as error:
        print(f"Error processing comment: {error}")


@app.get("/auth/instagram/callback")
async def instagram_callback(code: str | None = None):
    return {
        "message": "Instagram authorization callback received",
        "code_received": bool(code)
    }