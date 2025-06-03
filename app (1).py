import gradio as gr
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------
# Load smaller, valid models
# ------------------------------------------------------------------------

# Zero-shot NLI classifier (≈120 MB)
intent_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1"
)

# Lightweight sentence‐embedding model (≈60 MB)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")


# ------------------------------------------------------------------------
# Define intent labels
# ------------------------------------------------------------------------
intents = [
    "schedule appointment",
    "clinic hours",
    "insurance inquiry",
    "test results",
    "appointment cancellation",
    "prescription refill",
    "covid protocols",
    "update personal info",
    "services offered",
    "telehealth availability",
    "contact support"
]


# ------------------------------------------------------------------------
# Updated FAQ data and corresponding intent tags
# ------------------------------------------------------------------------
faq_data = {
    "questions": [
        "How do I schedule an appointment?",
        "Can I book an appointment online?",
        "What is the process to make a doctor's appointment?",
        "What are your clinic hours?",
        "Are you open on weekends?",
        "When does the clinic open and close?",
        "Do you accept my insurance?",
        "Which insurance plans do you accept?",
        "Can I use my insurance for treatment?",
        "How can I get my test results?",
        "When will my lab results be ready?",
        "How do I access my blood test results?",
        "How do I cancel or reschedule my appointment?",
        "Can I change my appointment time?",
        "What is the cancellation policy?",
        "How do I refill my prescription?",
        "Can I get a refill for my medication online?",
        "What is the process to renew my prescription?",
        "What COVID-19 protocols are in place?",
        "Do I need to wear a mask to my appointment?",
        "How is your clinic handling COVID-19 safety?",
        "How do I update my personal information?",
        "Can I change my address or phone number online?",
        "Where do I update my contact details?",
        "What services do you offer?",
        "Do you provide pediatric care?",
        "What types of medical services are available?",
        "Are telehealth visits available?",
        "Can I book a virtual doctor visit?",
        "Do you offer video consultations?",
        "How can I contact customer service?",
        "What is your support phone number?",
        "Who do I talk to for more help?"
    ],
    "answers": [
        "You can schedule an appointment by calling our clinic or using our online booking system.",
        "Yes, appointments can be booked online via our website or patient portal.",
        "To make a doctor's appointment, call our front desk or use the online scheduler.",
        "Our clinic is open Monday to Friday from 8 AM to 6 PM, and Saturdays from 9 AM to 1 PM.",
        "Yes, we are open on Saturdays from 9 AM to 1 PM but closed on Sundays.",
        "We open at 8 AM and close at 6 PM on weekdays.",
        "We accept most major insurance providers. Please contact us to verify your coverage.",
        "Our clinic accepts a variety of insurance plans including Aetna, Blue Cross, and United Healthcare.",
        "Yes, your insurance can be used for most treatments at our clinic.",
        "Test results are usually available within 3-5 business days and can be accessed through your patient portal.",
        "Lab results are typically ready within 3-5 days after testing.",
        "You can view your blood test results online through the patient portal.",
        "To cancel or reschedule, please call us at least 24 hours before your appointment or use the patient portal.",
        "You can change your appointment time by contacting our front desk.",
        "Our cancellation policy requires 24-hour notice to avoid fees.",
        "You can request prescription refills by contacting our pharmacy or through the patient portal.",
        "Refills can be requested online via your patient account.",
        "To renew your prescription, contact your doctor or use the online refill service.",
        "We follow all recommended COVID-19 safety protocols, including mask mandates and social distancing.",
        "Masks are required for all visitors during appointments.",
        "Our clinic enforces strict COVID-19 guidelines to ensure patient safety.",
        "You can update your personal information by logging into your patient account or contacting our front desk.",
        "Changes to your address or phone number can be made online.",
        "Please update your contact details through the patient portal or by calling us.",
        "Our services include primary care, pediatrics, lab testing, immunizations, and wellness checkups.",
        "Yes, we offer pediatric care as part of our services.",
        "We provide a wide range of medical services including preventive care and diagnostics.",
        "Telehealth visits are available for many of our services.",
        "You can book a virtual doctor visit online through our patient portal.",
        "Video consultations are offered for appropriate medical concerns.",
        "You can contact customer service via phone at 1-800-123-4567 or email support@clinic.com.",
        "Our support phone number is 1-800-123-4567.",
        "For more help, please contact our support team by phone or email."
    ]
}

intent_tags = [
    "schedule appointment", "schedule appointment", "schedule appointment",
    "clinic hours", "clinic hours", "clinic hours",
    "insurance inquiry", "insurance inquiry", "insurance inquiry",
    "test results", "test results", "test results",
    "appointment cancellation", "appointment cancellation", "appointment cancellation",
    "prescription refill", "prescription refill", "prescription refill",
    "covid protocols", "covid protocols", "covid protocols",
    "update personal info", "update personal info", "update personal info",
    "services offered", "services offered", "services offered",
    "telehealth availability", "telehealth availability", "telehealth availability",
    "contact support", "contact support", "contact support"
]


# ------------------------------------------------------------------------
# Precompute embeddings for all FAQ questions once
# ------------------------------------------------------------------------
faq_embeddings = model.encode(faq_data["questions"])


def chatbot_response(user_question, threshold_intent=0.2, threshold_faq=0.3):
    """
    Two‐stage matching:
      1) Zero‐shot intent classification. If confidence < threshold_intent,
         do a direct cosine‐similarity search across all FAQs.
      2) Otherwise, filter FAQs by predicted intent, then cosine‐similarity
         among that subset. If score < threshold_faq, return fallback.
    """
    # 1) Encode the user question
    user_emb = model.encode([user_question])[0]

    # 2) Intent prediction
    intent_result = intent_classifier(user_question, candidate_labels=intents)
    predicted_intent = intent_result["labels"][0]
    intent_confidence = intent_result["scores"][0]

    # 3) If low confidence, match across all FAQs
    if intent_confidence < threshold_intent:
        cos_sims = np.dot(faq_embeddings, user_emb) / (
            np.linalg.norm(faq_embeddings, axis=1) * np.linalg.norm(user_emb)
        )
        best_idx = np.argmax(cos_sims)
        if cos_sims[best_idx] >= threshold_faq:
            return faq_data["answers"][best_idx]
        return "Sorry, I didn't understand your question. Please try rephrasing."

    # 4) Otherwise, filter by that intent
    indices = [i for i, tag in enumerate(intent_tags) if tag == predicted_intent]
    if not indices:
        return "Sorry, I don't have an answer for that topic yet."

    filtered_embs = faq_embeddings[indices]
    cos_sims = np.dot(filtered_embs, user_emb) / (
        np.linalg.norm(filtered_embs, axis=1) * np.linalg.norm(user_emb)
    )
    best_idx_within = np.argmax(cos_sims)
    if cos_sims[best_idx_within] < threshold_faq:
        return "Sorry, I couldn't find a good match for your question."
    return faq_data["answers"][indices[best_idx_within]]


# ------------------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------------------
def respond(user_input):
    return chatbot_response(user_input)


iface = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(lines=2, placeholder="Ask a health-related question..."),
    outputs="text",
    title="Healthcare Support Chatbot",
    description="Ask about appointments, test results, COVID protocols, telehealth, and more."
)

if __name__ == "__main__":
    iface.launch()
