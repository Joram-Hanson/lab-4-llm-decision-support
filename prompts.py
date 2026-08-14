"""
Final prompt templates for the loan decision-support system.

Evolution notes:
- Summarization started as a bare "Summarize this:" instruction (V1), which produced
  inconsistent formatting between runs and occasionally added unsupported interpretive
  framing (e.g. turning a hedged statement into one of confidence). V2 added an explicit
  role, factuality constraints, a fixed length, and temperature=0 to fix both issues.
- Extraction was built directly with an explicit JSON schema, a one-shot example using
  a letter not in the dataset (to avoid contaminating later accuracy evaluation), and an
  explicit "use null, do not guess" rule, run at temperature=0 for reproducibility.
- The decision-support brief combines the letter and the extracted JSON, and explicitly
  forbids "approve"/"reject" recommendations, keeping the human loan officer as the final
  decision-maker.
"""

# Summarization

SUMMARY_PROMPT_V1 = "Summarize this:"

SUMMARY_SYSTEM_PROMPT = (
    "You are an assistant to a microfinance loan officer. Summarize loan application "
    "letters factually and neutrally. Do not invent details that are not stated in the "
    "letter. Keep the summary to 3-4 sentences."
)

def summary_prompt_v2(letter_text):
    return f"Summarize this loan application:\n\n{letter_text}"


#  Structured extraction 

EXTRACT_SYSTEM = (
    "You are a data extraction assistant for a microfinance loan officer. "
    "Extract structured data from loan application letters. "
    "Return ONLY a JSON object with exactly these keys: "
    "applicant_name (string), amount_ghs (number), purpose (string), "
    "monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean), "
    "repayment_months (number or null). "
    "If a field is not stated in the letter, use null. Do not guess. "
    "Return raw JSON only, no markdown formatting, no code fences, no extra text."
)

FEWSHOT_LETTER = """Dear Sir,
My name is Joram Hanson, a hairdresser in Tema. I request GHS 6,000 to buy new dryers.
My salon profit is about GHS 700 a month. I have no guarantor or collateral yet.
I hope to repay within 10 months."""

FEWSHOT_JSON = """{
  "applicant_name": "Joram Hanson",
  "amount_ghs": 6000,
  "purpose": "buy new dryers",
  "monthly_profit_ghs": 700,
  "has_collateral_or_guarantor": false,
  "repayment_months": 10
}"""

def extract_prompt(letter_text):
    return (
        f"Example letter:\n{FEWSHOT_LETTER}\n\n"
        f"Example output:\n{FEWSHOT_JSON}\n\n"
        f"Now extract from this letter:\n{letter_text}"
    )


#  Decision-support brief 

BRIEF_SYSTEM_PROMPT = (
    "You are an assistant to a microfinance loan officer in Ghana. Given a loan "
    "application letter and its extracted data, produce a decision-support brief with "
    "exactly these four sections:\n"
    "1. Strengths (bullet points, grounded only in the letter)\n"
    "2. Risks / red flags (bullet points)\n"
    "3. Missing information the officer should request\n"
    "4. Suggested next step (one of: 'invite for interview', 'request documents', "
    "'flag for senior review' — never 'approve' or 'reject')\n"
    "The final lending decision is always made by a human loan officer. You must never "
    "recommend approving or rejecting the loan."
)

def brief_prompt(letter_text, extracted_json):
    import json
    return (
        f"Loan application letter:\n{letter_text}\n\n"
        f"Extracted data:\n{json.dumps(extracted_json, indent=2)}\n\n"
        f"Produce the decision-support brief."
    )