# AI Learning Hub — Instagram RAG Chatbot

This is my assignment project — a chatbot that auto-replies to people on
Instagram, either when they DM the account or when they comment on a post.
It's built with a small RAG pipeline on top of Groq (LLM) + FAISS
(vector search), wired up to Instagram through Meta's Graph API and
webhooks.

## What it actually does

- Someone DMs the business Instagram account → bot replies automatically.
- Someone comments on a post → bot replies publicly under the comment.
- Answers only come from `knowledge_base/faq.txt` — if something isn't in
  there, the bot says it doesn't know instead of making stuff up.
- It remembers the last few messages in a DM conversation, so you can ask
  a follow-up like "how much does it cost?" after asking about a specific
  course, and it'll figure out which course you meant.

## How it's put together

```
app/
  config.py              -> loads everything from .env
  router.py               -> figures out intent (pricing, enrollment, etc.)
  retrieval.py             -> FAISS similarity search over the knowledge base
  rag.py                   -> ties router + retrieval + Groq together, builds the answer
  memory.py                -> keeps last 6 messages per user for context
  load_documents.py        -> reads .txt files from knowledge_base/
  chunk_documents.py       -> splits documents into chunks for embedding
  create_vectorstore.py    -> builds the FAISS index (run once, or whenever faq.txt changes)
  instagram/
    webhook.py              -> the FastAPI server Meta sends events to
    sender.py                -> functions that actually call the Instagram API to send replies
knowledge_base/
  faq.txt                   -> the only source of truth the bot is allowed to use
vectorstore/
  index.faiss / index.pkl   -> the built vector index (generated file, not hand-written)
```

Intent classification is a hybrid: fast keyword rules first (`router.py`),
and only if that's not confident enough does it call the LLM to classify.
Saves a lot of unnecessary Groq calls for obvious stuff like "what is the
price" type questions.

## Running it locally

```bash
pip install -r requirements.txt
python app/create_vectorstore.py
uvicorn app.instagram.webhook:app --host 0.0.0.0 --port 8000
```

You'll need a way to expose port 8000 to the internet since Meta needs a
public HTTPS URL to send webhooks to — I used ngrok for this
(`ngrok http 8000`).

## Setting it up on the Meta side (the part that actually took the most time)

1. Create an app on developers.facebook.com, add the Instagram product.
2. Under Webhooks, set the callback URL to `https://<your-ngrok-url>/webhook`
   and put in whatever you set `META_VERIFY_TOKEN` to.
3. Subscribe to the `messages` and `comments` fields (and `live_comments`
   if you care about live streams).
4. Under App Roles → Roles, add whichever personal Instagram account
   you're going to test with as an **Instagram Tester** (not the plain
   "Tester" role — that one's different and won't work for this).
5. On that personal account, open Instagram → Edit Profile → Apps and
   Websites → Tester Invites tab, and accept the invite. This part isn't
   always in the same spot depending on the app version — on desktop it
   was under instagram.com/accounts/manage_access, not in the mobile
   app's newer settings menu.
6. **Publish the app.** This is the step that actually got things working
   for me. Even with the webhook configured correctly and the tester
   invite accepted, nothing was reaching my server until I went to the
   "Publish" page in the dashboard and clicked Publish. Apparently Meta
   just won't send webhook events at all if the app is still sitting in
   Development mode — this isn't obvious from the webhook setup docs and
   cost me a while to figure out. You'll need a Privacy Policy URL and an
   app icon filled in under App Settings → Basic before it'll let you
   publish (I just used my GitHub repo link for the privacy policy since
   this isn't a real public product).


## Known limitation

The knowledge base has both course fees in one paragraph
(`knowledge_base/faq.txt`), so if you ask "how much does it cost" after
asking about one specific course, the bot sometimes answers with both
courses' fees instead of just the one you meant, since they land in the
same retrieved chunk. Not a code bug — just something to fix by splitting
that section into two separate paragraphs if I get time before submitting.