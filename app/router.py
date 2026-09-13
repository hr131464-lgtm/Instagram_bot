import json
import re

from groq import Groq

from app.config import GROQ_API_KEY


# ============================================================
# CONFIGURATION
# ============================================================

INTENTS = {
    "PRICING",
    "COURSE_INFO",
    "DURATION",
    "REFUND",
    "INTERNSHIP",
    "CERTIFICATE",
    "ONLINE_CLASSES",
    "SUPPORT",
    "ENROLLMENT",
    "HUMAN_HANDOFF",
    "GENERAL_FAQ",
    "UNKNOWN",
}

LLM_MODEL = "openai/gpt-oss-120b"

# If rule confidence is below this value,
# the query is sent to the LLM.
RULE_CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# RULE-BASED INTENT KEYWORDS
# ============================================================

RULES = {
    "PRICING": [
        "price",
        "pricing",
        "fee",
        "fees",
        "cost",
        "how much",
        "₹",
        "rupees",
        "rs",
        "payment",
        "pay",
        "charge",
        "charges",
    ],

    "DURATION": [
        "duration",
        "how long",
        "weeks",
        "week",
        "months",
        "month",
        "length of the course",
        "course length",
        "how many weeks",
        "how many months",
    ],

    "REFUND": [
        "refund",
        "refunds",
        "money back",
        "get my money back",
        "cancel enrollment",
        "cancellation",
        "cancel my enrollment",
    ],

    "INTERNSHIP": [
        "internship",
        "internships",
        "intern",
        "internship opportunity",
        "internship opportunities",
    ],

    "CERTIFICATE": [
        "certificate",
        "certificates",
        "certification",
        "completion certificate",
        "course completion certificate",
    ],

    "ONLINE_CLASSES": [
        "online",
        "remote",
        "online class",
        "online classes",
        "online course",
        "study remotely",
        "attend remotely",
    ],

    "SUPPORT": [
        "support",
        "contact support",
        "customer support",
        "support team",
        "help desk",
        "contact",
        "help",
    ],

    "ENROLLMENT": [
        "enroll",
        "enrollment",
        "join",
        "register",
        "registration",
        "admission",
        "sign up",
        "signup",
    ],

    "HUMAN_HANDOFF": [
        "human",
        "agent",
        "representative",
        "talk to someone",
        "talk to a person",
        "talk to human",
        "speak to someone",
        "speak to a person",
        "real person",
        "customer care",
        "customer service",
    ],

    "COURSE_INFO": [
        "course",
        "courses",
        "learn",
        "learning",
        "curriculum",
        "syllabus",
        "subject",
        "subjects",
        "topics",
        "what do you teach",
        "what is covered",
        "what will i learn",
        "what can i learn",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize user input before classification.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# KEYWORD MATCHING
# ============================================================

def keyword_matches(text: str, keyword: str) -> bool:
    """
    Safely check whether a keyword exists in the text.

    Multi-word phrases are matched directly.

    Single words use word boundaries so that short words
    such as 'rs' do not accidentally match inside words
    like 'course'.
    """

    keyword = keyword.lower().strip()

    if not keyword:
        return False

    # Special currency symbol
    if keyword == "₹":
        return "₹" in text

    # Multi-word phrase
    if " " in keyword:
        return keyword in text

    # Single word / token
    pattern = rf"\b{re.escape(keyword)}\b"

    return bool(re.search(pattern, text))


# ============================================================
# RULE-BASED CLASSIFIER
# ============================================================

def rule_based_classification(text: str):
    """
    Fast first-stage intent classification.

    High-confidence queries are handled by rules.
    Ambiguous queries are sent to the LLM.
    """

    text = normalize_text(text)

    if not text:
        return {
            "intent": "UNKNOWN",
            "confidence": 0.0,
            "method": "rule_based",
            "use_rag": False,
            "requires_human": False,
        }

    scores = {}

    for intent, keywords in RULES.items():

        score = 0

        for keyword in keywords:

            if keyword_matches(text, keyword):

                # Strong pricing signals
                if intent == "PRICING":

                    if " " in keyword:
                        score += 4
                    else:
                        score += 3

                # Multi-word phrases are stronger
                elif " " in keyword:
                    score += 2

                # Normal single-word signal
                else:
                    score += 1

        if score > 0:
            scores[intent] = score

    # --------------------------------------------------------
    # No rule matched
    # --------------------------------------------------------

    if not scores:
        return {
            "intent": "GENERAL_FAQ",
            "confidence": 0.50,
            "method": "rule_based",
            "use_rag": True,
            "requires_human": False,
        }

    # --------------------------------------------------------
    # Intent priority
    # --------------------------------------------------------

    priority = [
        "HUMAN_HANDOFF",
        "PRICING",
        "REFUND",
        "ENROLLMENT",
        "INTERNSHIP",
        "CERTIFICATE",
        "DURATION",
        "ONLINE_CLASSES",
        "SUPPORT",
        "COURSE_INFO",
    ]

    best_intent = max(
        scores,
        key=lambda intent: (
            scores[intent],
            -priority.index(intent),
        ),
    )

    matched_score = scores[best_intent]

    # Convert rule score into confidence.
    confidence = min(
        0.60 + (matched_score * 0.10),
        0.95,
    )

    return {
        "intent": best_intent,
        "confidence": round(confidence, 2),
        "method": "rule_based",
        "use_rag": best_intent not in {
            "HUMAN_HANDOFF",
            "ENROLLMENT",
        },
        "requires_human": best_intent == "HUMAN_HANDOFF",
    }


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str):
    """
    Extract a JSON object from an LLM response.

    Handles:
    - normal JSON
    - markdown code fences
    - JSON embedded in additional text
    - invalid responses
    """

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("```", "").strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # JSON embedded inside text
    # --------------------------------------------------------

    match = re.search(
        r"\{.*?\}",
        text,
        re.DOTALL,
    )

    if match:

        try:
            return json.loads(match.group(0))

        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# LLM INTENT CLASSIFIER
# ============================================================

def llm_classify_intent(text: str):
    """
    Second-stage LLM classifier.

    This is used only when the rule-based classifier
    is uncertain.
    """

    if not GROQ_API_KEY:

        return {
            "intent": "GENERAL_FAQ",
            "confidence": 0.0,
            "method": "llm_unavailable",
            "use_rag": True,
            "requires_human": False,
        }

    client = Groq(
        api_key=GROQ_API_KEY
    )

    prompt = f"""
Classify the user's message into exactly ONE intent.

Allowed intents:

PRICING
COURSE_INFO
DURATION
REFUND
INTERNSHIP
CERTIFICATE
ONLINE_CLASSES
SUPPORT
ENROLLMENT
HUMAN_HANDOFF
GENERAL_FAQ
UNKNOWN

Intent definitions:

PRICING:
Questions about fees, prices, cost, payment amount,
or how much a course costs.

COURSE_INFO:
Questions about courses, curriculum, syllabus,
subjects, topics, or what is taught.

DURATION:
Questions about course length, weeks, months,
or how long a course takes.

REFUND:
Questions about refunds, cancellations,
or getting money back.

INTERNSHIP:
Questions about internships or internship opportunities.

CERTIFICATE:
Questions about certificates or course completion certificates.

ONLINE_CLASSES:
Questions about online or remote classes.

SUPPORT:
Questions about contacting or getting help from support.

ENROLLMENT:
Requests to enroll, register, join, sign up,
or get admission.

HUMAN_HANDOFF:
Requests to speak with a human, agent,
representative, customer care, or real person.

GENERAL_FAQ:
General questions that may potentially be answered
from the knowledge base.

UNKNOWN:
The message does not fit any useful category.

Important classification rules:

1. "fee", "price", "cost", "payment", "how much"
   normally means PRICING.

2. If a question asks about both a course and its price,
   choose PRICING.

3. "course", "learning", or "learn" alone means
   COURSE_INFO.

4. Questions about weeks, months, duration, or length
   mean DURATION.

5. Requests to speak with a real person mean
   HUMAN_HANDOFF.

6. Requests to register, enroll, join, or sign up mean
   ENROLLMENT.

7. Return ONLY valid JSON.

8. Do not use markdown.

9. Do not add explanations.

User message:
{text}

Return exactly:

{{"intent":"GENERAL_FAQ","confidence":0.95}}
"""

    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict intent classification "
                        "system. Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        # ----------------------------------------------------
        # Empty response
        # ----------------------------------------------------

        if not content:
            raise ValueError(
                "LLM returned an empty response."
            )

        # ----------------------------------------------------
        # Extract JSON
        # ----------------------------------------------------

        result = extract_json(content)

        if not result:
            raise ValueError(
                "Could not extract valid JSON from "
                f"LLM response: {content}"
            )

        # ----------------------------------------------------
        # Validate intent
        # ----------------------------------------------------

        intent = result.get(
            "intent",
            "GENERAL_FAQ",
        )

        if intent not in INTENTS:
            intent = "GENERAL_FAQ"

        # ----------------------------------------------------
        # Validate confidence
        # ----------------------------------------------------

        try:
            confidence = float(
                result.get(
                    "confidence",
                    0.0,
                )
            )

        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        return {
            "intent": intent,
            "confidence": round(confidence, 2),
            "method": "llm",
            "use_rag": intent not in {
                "HUMAN_HANDOFF",
                "ENROLLMENT",
            },
            "requires_human": intent == "HUMAN_HANDOFF",
        }

    except Exception as error:

        print(
            f"LLM intent classification failed: {error}"
        )

        # Safe fallback.
        return {
            "intent": "GENERAL_FAQ",
            "confidence": 0.0,
            "method": "llm_fallback",
            "use_rag": True,
            "requires_human": False,
        }


# ============================================================
# HYBRID CLASSIFIER
# ============================================================

def classify_intent(text: str):
    """
    Hybrid classification pipeline.

    Stage 1:
        Fast rule-based classifier.

    Stage 2:
        LLM classifier for uncertain queries.
    """

    rule_result = rule_based_classification(text)

    # High-confidence rule result
    if (
        rule_result["confidence"]
        >= RULE_CONFIDENCE_THRESHOLD
    ):
        return rule_result

    # Low-confidence / ambiguous query
    return llm_classify_intent(text)


# ============================================================
# TEST SUITE
# ============================================================

if __name__ == "__main__":

    test_questions = [

        # ----------------------------------------------------
        # PRICING
        # ----------------------------------------------------

        "How much does the Generative AI course cost?",
        "What is the fee for learning Generative AI?",
        "How much is the course?",
        "What is the price of the AI course?",

        # ----------------------------------------------------
        # DURATION
        # ----------------------------------------------------

        "How long is the course?",
        "How many weeks is the beginner course?",

        # ----------------------------------------------------
        # REFUND
        # ----------------------------------------------------

        "Do you provide refunds?",
        "Can I get my money back?",

        # ----------------------------------------------------
        # INTERNSHIP
        # ----------------------------------------------------

        "Do you offer internships?",
        "Are internship opportunities available?",

        # ----------------------------------------------------
        # CERTIFICATE
        # ----------------------------------------------------

        "Will I get a certificate?",
        "Do you provide course completion certificates?",

        # ----------------------------------------------------
        # ONLINE CLASSES
        # ----------------------------------------------------

        "Are classes online?",
        "Can I attend classes remotely?",

        # ----------------------------------------------------
        # SUPPORT
        # ----------------------------------------------------

        "How can I contact support?",
        "I need help from the support team.",

        # ----------------------------------------------------
        # ENROLLMENT
        # ----------------------------------------------------

        "I want to enroll",
        "How can I register for the course?",

        # ----------------------------------------------------
        # HUMAN HANDOFF
        # ----------------------------------------------------

        "I want to talk to a human",
        "Can I speak to a real person?",

        # ----------------------------------------------------
        # COURSE INFORMATION
        # ----------------------------------------------------

        "What courses do you offer?",
        "What will I learn in the Generative AI course?",

        # ----------------------------------------------------
        # AMBIGUOUS / UNKNOWN
        # These should normally go through the LLM.
        # ----------------------------------------------------

        "Can students stay somewhere near the institute?",
        "Do you have accommodation?",
        "Is there somewhere I can live while studying?",
    ]

    print("=" * 60)
    print("HYBRID INTENT ROUTER TEST")
    print("=" * 60)

    for question in test_questions:

        result = classify_intent(question)

        print("\nQuestion:", question)
        print("Intent:", result["intent"])
        print("Confidence:", result["confidence"])
        print("Method:", result["method"])
        print("Use RAG:", result["use_rag"])
        print(
            "Requires human:",
            result["requires_human"],
        )