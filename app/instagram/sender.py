import requests

from app.config import (
    META_GRAPH_API_VERSION,
    META_INSTAGRAM_ACCESS_TOKEN,
    META_INSTAGRAM_ACCOUNT_ID,
)


def send_instagram_message(recipient_id, message):
    if not META_INSTAGRAM_ACCESS_TOKEN:
        raise ValueError(
            "META_INSTAGRAM_ACCESS_TOKEN is missing."
        )

    if not META_INSTAGRAM_ACCOUNT_ID:
        raise ValueError(
            "META_INSTAGRAM_ACCOUNT_ID is missing."
        )

    url = (
        f"https://graph.instagram.com/"
        f"{META_GRAPH_API_VERSION}/"
        f"{META_INSTAGRAM_ACCOUNT_ID}/messages"
    )

    payload = {
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message
        },
        "access_token": META_INSTAGRAM_ACCESS_TOKEN
    }

    response = requests.post(
        url,
        json=payload,
        timeout=15
    )

    if not response.ok:
        raise RuntimeError(
            f"Instagram API error: "
            f"{response.status_code} - {response.text}"
        )

    return response.json()


def reply_to_comment(comment_id, message):
    """
    Post a PUBLIC reply underneath a comment on one of our posts.

    Uses POST /{comment-id}/replies, which is how the Instagram
    Graph API implements "reply to this comment" (it shows up
    nested under the original comment, visible to everyone).
    """

    if not META_INSTAGRAM_ACCESS_TOKEN:
        raise ValueError(
            "META_INSTAGRAM_ACCESS_TOKEN is missing."
        )

    url = (
        f"https://graph.instagram.com/"
        f"{META_GRAPH_API_VERSION}/"
        f"{comment_id}/replies"
    )

    payload = {
        "message": message,
        "access_token": META_INSTAGRAM_ACCESS_TOKEN
    }

    response = requests.post(
        url,
        data=payload,
        timeout=15
    )

    if not response.ok:
        raise RuntimeError(
            f"Instagram API error (comment reply): "
            f"{response.status_code} - {response.text}"
        )

    return response.json()