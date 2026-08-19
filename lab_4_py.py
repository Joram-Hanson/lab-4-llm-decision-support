# %% [markdown]
# # Lab 4: LLMs and Prompt Engineering for Decision Support
# 
# **Duration:** 2 weeks [30 Jul - 13 Aug, 2026]
# **Due Date:** 13th August, 2026
# **Format:** Jupyter Notebook / Google Colab + external APIs + GitHub version control
# **Grading:** This is a graded lab.
# 
# **Student Name:** Joram Hanson
# **Student ID:** 25432028
# 
# ---
# 
# ### Objective
# 
# In the previous labs you *trained* models. In this lab you will *use* a model that someone
# else spent millions of dollars training — a **Large Language Model (LLM)** — and learn that
# getting good results out of one is an engineering discipline of its own: **prompt
# engineering**.
# 
# You will build a **decision support system for a microfinance loan officer**. Given a pile of
# free-text loan application letters, your system will:
# 
# 1. **Summarize** each application into a short, factual brief,
# 2. **Extract** specific structured data points (JSON) that a downstream system could store,
# 3. Produce a **decision-support recommendation** — while keeping the human firmly in the loop.
# 
# Just as importantly, you will **evaluate** the LLM's output for quality, reliability, and
# appropriateness: Does it hallucinate? Is it consistent across runs? Should it be trusted to
# make the final call?
# 
# ---
# 
# ### Choosing an API provider
# 
# You need an LLM API with a **free tier**. Recommended options (pick ONE):
# 
# | Provider | Free tier | Notes |
# |---|---|---|
# | **Groq** (recommended) | Yes, generous | OpenAI-compatible API, very fast, open models (Llama) |
# | **Google Gemini** | Yes | `google-generativeai` package |
# | **Hugging Face Inference API** | Yes, limited | Many open models |
# | OpenAI / Anthropic | Paid | Fine if you already have credits |
# 
# The notebook's example code uses the **OpenAI-compatible chat format** (works with Groq and
# OpenAI directly; Gemini users adapt the call in one place). Everything else in the lab is
# provider-agnostic.

# %% [markdown]
# ---
# ### Part 0: Repository and API-key setup
# 
# 1. Create a **public** repository named `lab-4-llm-decision-support` and save this notebook
#    inside it.
# 2. Sign up with your chosen provider and create an **API key**.
# 3. **NEVER hard-code or commit your API key.** This is a graded requirement.
#    - Locally: put it in a `.env` file and add `.env` to `.gitignore`.
#    - Colab: use the Secrets panel (key icon) and read it with `google.colab.userdata`.
# 4. Add a `requirements.txt`: `openai python-dotenv pandas matplotlib`.
# 5. Commit and push after **each Part** — we will check for incremental commits.
# 
# > **A leaked key in your commit history = resubmission + penalty.** Keys can be scraped from
# > public repos within minutes.

# %%
# API-key setup — DO NOT hard-code your key in this cell.

import os
from dotenv import load_dotenv

# --- Local (with a .env file) ---
# from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ["GROQ_API_KEY"]

# --- Google Colab (Secrets panel) ---
# from google.colab import userdata
# API_KEY = userdata.get("GROQ_API_KEY")

# TODO: set API_KEY using ONE of the methods above.

# OpenAI-compatible client (works for Groq and OpenAI; Gemini users see their docs):
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",   # remove this line if using OpenAI itself
)
MODEL = "llama-3.3-70b-versatile"                # or your provider's model name

print("Client ready.")

# %% [markdown]
# ---
# # Section 1 — Talking to an LLM Programmatically
# 
# Before building anything, understand the anatomy of an API call: **messages and roles**
# (`system`, `user`, `assistant`), and the **generation parameters** (`temperature`,
# `max_tokens`).

# %% [markdown]
# ### Part 1.1 — Your first API call

# %%
# TODO: Write a helper function you will reuse for the WHOLE lab:
#
def ask_llm(user_prompt, system_prompt="You are a helpful assistant.",
            temperature=0.7, max_tokens=500):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
answer = ask_llm("What is the capital of Ghana, and roughly what is its population?")
print(answer)



# TODO: Call it once with a simple question and print the answer.
# TODO: Print response.usage as well — how many tokens did your call consume?

# %%
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Ghana, and roughly what is its population?"},
    ],
)
print(response.usage)

# %% [markdown]
# **Student Reasoning — Anatomy of a call**
# *1. What is the difference between the `system` and `user` roles? Give an example of
# something that belongs in each.*
# *2. What is a token, roughly? Why do API providers bill per token rather than per request?*
# 
# > **Answer:** [Double-click to edit]
# 1. The system message tells the model its role and rules for the whole conversation. For example, "You are a support agent for a banking app, never ask for a PIN." whiles the user message is the actual question or task for that turn, for example, "My transfer failed, what do I do?"
# 2. A token is roughly a chunk of text,often a word, part of a word which in my test question used 55 prompt tokens and I got 51 token answer for a total of 106. And also Providers charge per token because token count reflects actual compute cost, so a short request and a long request are billed fairly instead of charging the same flat fee for both.

# %% [markdown]
# ### Part 1.2 — Temperature: the randomness dial

# %%
# TODO: Ask the SAME question 5 times at temperature=0.0 and 5 times at temperature=1.2.
#   A good test question: "Suggest a name for a savings product for market traders in Accra."
question = "Suggest a name for a savings product for market traders in Accra."
low_temp_answers = [ask_llm(question, temperature=0.0) for _ in range(5)]
high_temp_answers = [ask_llm(question, temperature=1.2) for _ in range(5)]

# TODO: Print all 10 answers, grouped by temperature.
print("Low temperature answers(0.0):")
for i,answer in enumerate(low_temp_answers, 1):
    print(f"{i}: {answer}\n")

print("High temperature answers(1.2):")
for i,answer in enumerate(high_temp_answers, 1):
    print(f"{i}: {answer}\n")

# %% [markdown]
# **Student Reasoning — Temperature**
# *What did you observe at each temperature? For the loan decision-support system you are about
# to build, which temperature regime is appropriate, and why?*
# 
# > **Answer:** [Double-click to edit]
# At temperature 0.0, the model produced fairly consistent answers across runs. Names like Makola Save and Trader's Treasure repeated in most of the five runs, with only small variations elsewhere. 
# At temperature 1.2, the answers varied much more. Different names appeared almost every run, different local languages were used inconsistently, and one word, Kokroko, was even given three different meanings across the runs. For the loan decision-support system, temperature 0.0 is the appropriate choice. The system needs to extract exact loan amounts and produce reliable recommendations every time, and the high temperature runs showed the model can become inconsistent or even invent conflicting facts, which is too risky for a system meant to support real financial decisions.
# 
# 

# %% [markdown]
# ---
# # Section 2 — The Dataset: Loan Application Letters
# 
# Run the next cell to load **six loan application letters** submitted to a (fictional)
# microfinance institution in Ghana, plus **gold-standard extraction labels** for three of them
# (you will use these for evaluation in Section 4).
# 
# Read at least two letters fully before moving on — you cannot engineer prompts for text you
# have not read.

# %%
LETTERS = {
"L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",

"L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",

"L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",

"L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",

"L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",

"L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
  "L001": {"applicant_name": "Akosua Mensah", "amount_ghs": 8000,  "purpose": "buy deep freezer / expand into frozen foods",
           "monthly_profit_ghs": 900,  "has_collateral_or_guarantor": True,  "repayment_months": 20},
  "L003": {"applicant_name": "Efua Darko",    "amount_ghs": 15000, "purpose": "industrial sewing machines and fabric stock",
           "monthly_profit_ghs": 2800, "has_collateral_or_guarantor": True,  "repayment_months": 15},
  "L006": {"applicant_name": "Kofi",          "amount_ghs": 50000, "purpose": "car wash, provision shop, phone imports",
           "monthly_profit_ghs": None, "has_collateral_or_guarantor": False, "repayment_months": 12},
}

print(f"{len(LETTERS)} letters loaded.")

# %% [markdown]
# ---
# # Section 3 — Prompt Engineering for the Decision Support System
# 
# You will now build the three components of the system, iterating on your prompts as you go.
# **Keep every major prompt version** — Section 3.4 asks you to commit your prompt templates
# and document how they evolved.

# %% [markdown]
# ### Part 3.1 — Component 1: Summarization
# Turn a rambling letter into a 3-4 sentence factual brief a busy loan officer can scan.

# %%
# TODO: Write SUMMARY_PROMPT_V1 — your first, naive attempt (e.g. just "Summarize this:").
#   Run it on L002 and L006. Read the output critically.
SUMMARY_PROMPT_V1 = "Summarize this loan application: "

for letter_id in ["L002", "L006"]:
    result = ask_llm(f"{SUMMARY_PROMPT_V1}\n\n{LETTERS[letter_id]}")
    print(f"Summary for {letter_id}:\n{result}\n")
    print()

# TODO: Now write SUMMARY_PROMPT_V2 as a proper template with:
#   - a system prompt giving the LLM a ROLE (e.g. "You are an assistant to a microfinance
#     loan officer...") and constraints (factual, neutral, no invented details, 3-4 sentences)
#   - a user prompt template like: f"Summarize this loan application:\n\n{letter_text}"
#   Run V2 on the same two letters at temperature=0.
SUMMARY_PROMPT_V2_SYSTEM = (
    "You are an assistant to a microfinance loan officer. "
    "Summarize loan applications factually and neutrally, in 3-4 sentences. "
    "Do not invent details."
)

def summaary_prompt_v2(letter_text):
    return f"Summarize this loan application:\n\n{letter_text}"

for letter_id in ["L002", "L006"]:
    result = ask_llm(summaary_prompt_v2(LETTERS[letter_id]), system_prompt=SUMMARY_PROMPT_V2_SYSTEM, temperature=0.0)
    print(f"Summary for {letter_id} (V2):\n{result}\n")
    print()

# TODO: Compare V1 vs V2 outputs side by side. Keep both prompt versions in this notebook.
for letter_id in ["L002", "L006"]:
    v1_result = ask_llm(f"{SUMMARY_PROMPT_V1}\n\n{LETTERS[letter_id]}")
    v2_result = ask_llm(summaary_prompt_v2(LETTERS[letter_id]), system_prompt=SUMMARY_PROMPT_V2_SYSTEM, temperature=0.0)
    print(f"Letter {letter_id}:\nV1 Summary:\n{v1_result}\n\nV2 Summary:\n{v2_result}\n")

# %% [markdown]
# **Student Reasoning — Summarization prompts**
# *1. What concrete problems did V1's output have that V2 fixed? Quote examples.*
# *2. Why is "no invented details" an essential instruction in this application? What is this
# failure mode called in the LLM literature?*
# 
# > **Answer:** [Double-click to edit]
# 1. V1 was inconsistent between runs, one attempt gave a bulleted list for L006, another gave a plain paragraph, since V1 had no structure instruction and no temperature control. V2 stayed consistent (3-4 sentences, temperature=0) because its system prompt required it. V1 also added unsupported framing, calling Kwame "confident" he'd repay, when the letter only said he could pay back "whenever the money comes" and hoped for the best. V2 kept that uncertainty intact instead of inventing confidence.
# 
# 2. This matters because the loan officer relies on the summary to judge real risk. Invented details, like false confidence, could mislead that judgement. This failure mode is called "hallucination" in the LLM literature.

# %% [markdown]
# ### Part 3.2 — Component 2: Structured extraction (JSON)
# Downstream software cannot read prose. Extract the fields in `GOLD` as strict JSON.

# %%
# TODO: Write EXTRACT_PROMPT — a template that instructs the model to return ONLY a JSON
#   object with EXACTLY these keys:
#     applicant_name (string), amount_ghs (number), purpose (string),
#     monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean),
#     repayment_months (number or null)
#   Techniques to use:
#     - explicit schema in the prompt
#     - ONE worked example (few-shot) using a letter you write yourself (not from LETTERS!)
#     - "If a field is not stated in the letter, use null. Do not guess."
#     - temperature=0
import json
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

def extract_fields(letter_text, temperature=0):
    raw = ask_llm(extract_prompt(letter_text), system_prompt=EXTRACT_SYSTEM, temperature=temperature)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("json", "", 1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"WARNING: failed to parse JSON for letter. Raw output:\n{raw}")
        return None

import pandas as pd

rows = []
for letter_id, letter_text in LETTERS.items():
    fields = extract_fields(letter_text)
    if fields is not None:
        fields["letter_id"] = letter_id
        rows.append(fields)

extraction_df = pd.DataFrame(rows).set_index("letter_id")
extraction_df





# %%
#Running extraction on L002 using a version of the prompt with the "use null for missing fields" instruction removed, to see if the model will guess values.
EXTRACT_SYSTEM_NO_NULL = (
    "You are a data extraction assistant for a microfinance loan officer. "
    "Extract structured data from loan application letters. "
    "Return ONLY a JSON object with exactly these keys: "
    "applicant_name (string), amount_ghs (number), purpose (string), "
    "monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean), "
    "repayment_months (number or null). "
    "Return raw JSON only, no markdown formatting, no code fences, no extra text."
)

raw = ask_llm(extract_prompt(LETTERS["L002"]), system_prompt=EXTRACT_SYSTEM_NO_NULL, temperature=0)
print(f"Raw output for L002 with no 'use null' instruction:\n{raw}\n")

# %% [markdown]
# **Student Reasoning — Structured extraction**
# *1. Why must the few-shot example NOT come from the six letters you are processing?*
# *2. Why "use null, do not guess" — what did the model do without that instruction?*
# *3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?*
# 
# > **Answer:** [Double-click to edit]
# 1. The few-shot example has to come from outside the six letters because using one of them would let the model see the exact answer to something it's later graded on. That would turn the accuracy check in Section 4 into a memory test instead of a real measure of extraction ability.
# 
# 2. After removing the "use null, don't guess" line and running extraction on L002, which has no stated profit or repayment terms, the model still returned null for both fields instead of making something up. So this particular case didn't show fabrication. Even so, the instruction is worth keeping, because trusting the model to behave safely by default is risky, a different prompt or a different letter could easily produce an invented number instead. Spelling it out removes that guesswork.
# 
# 3. Extraction needs temperature=0 because each field has a single correct answer, and the output has to stay the same every time it runs, which lines up with what the temperature experiment already showed, low temperature sticks close to one stable result. Creative tasks are different since there's no single right answer, so some randomness there actually helps rather than hurts.

# %% [markdown]
# ### Part 3.3 — Component 3: The decision-support brief
# Combine everything: for each letter, produce a recommendation brief for the loan officer —
# strengths, risks, missing information, and a suggested next step. The system must
# **support** the decision, not **make** it.

# %%
# TODO: Write BRIEF_PROMPT — it receives the letter AND your extracted JSON, and must output:
#     1. Strengths (bullet points, grounded in the letter)
#     2. Risks / red flags (bullet points)
#     3. Missing information the officer should request
#     4. Suggested next step (e.g. "invite for interview", "request documents",
#        "flag for senior review") — NOT "approve" or "reject".
#   Give the model an explicit instruction that final decisions are made by humans.
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
    return (
        f"Loan application letter:\n{letter_text}\n\n"
        f"Extracted data:\n{json.dumps(extracted_json, indent=2)}\n\n"
        f"Produce the decision-support brief."
    )
# TODO: Generate briefs for ALL SIX letters. Print the briefs for L001, L002, and L006 —
#   three very different applications.
briefs = {}
for letter_id, letter_text in LETTERS.items():
    extracted = extraction_df.loc[letter_id].to_dict()
    brief = ask_llm(brief_prompt(letter_text, extracted), system_prompt=BRIEF_SYSTEM_PROMPT, temperature=0)
    briefs[letter_id] = brief

for letter_id in ["L001", "L002", "L006"]:
    print(f"{letter_id} ")
    print(briefs[letter_id])
    print()

# %%
print(briefs["L003"])

# %% [markdown]
# **Student Reasoning — Decision support**
# *1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the
# system identify the right strengths and red flags in each?*
# *2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and
# one ethical reason.*
# 
# > **Answer:** [Double-click to edit]
# 1. The system clearly told the two apart. For L003 (strong), it found real, checkable strengths, a registered business, a GHS 5,000 fixed deposit as collateral, solid revenue (GHS 22,000 in December, GHS 2,800 average monthly profit), and 18 months of sales records. For L006 (weak), the only "strength" was soft and unverifiable, just that his friends think he's business minded. The risks matched this pattern too, L003's were about cash flow tightness on an otherwise solid business, while L006's were fundamental, no experience, no collateral, an unproven repayment plan, and three unrelated ventures spreading his resources thin. So the system correctly picked up on the difference in quality.
# 
# One inconsistency though: both got the same next step, "invite for interview," even though L006 is clearly weaker. A sharper system might have flagged it for senior review instead.
# 
# 2. On the practical side, the model has no way to actually verify anything in these letters, no way to confirm someone's identity, check their credit history, catch fraud, or look at real documents. Letting it say "approve" or "reject" would mean handing out real money based purely on unverified claims in a letter, which is way too shaky a foundation. On the ethical side, letting the model make the final call takes accountability out of a decision that seriously affects someone's life and livelihood, and it opens the door to quiet bias, like penalizing someone who writes poorly in English even though their business is genuinely solid, with no human there to catch it or let the person appeal.

# %% [markdown]
# ### Part 3.4 — Commit your prompt templates
# Prompts ARE code. Save your final `SUMMARY_PROMPT`, `EXTRACT_PROMPT`, and `BRIEF_PROMPT` into
# a separate file `prompts.py` (or `prompts.md`) in your repository and commit it with a
# message describing how the prompts evolved. Paste your commit hash below.
# 
# > **Commit hash:** fcba71db50dd990e49b2b24d66857fb6da416727
# 

# %% [markdown]
# ---
# # Section 4 — Evaluation: Quality, Reliability, Appropriateness
# 
# An impressive demo is not a trustworthy system. Now measure it.

# %% [markdown]
# ### Part 4.1 — Extraction accuracy against gold labels

# %%


# %%
# TODO: For the three letters in GOLD, compare your extracted DataFrame to the gold values
#   field by field. Compute per-field accuracy across the three letters
#   (name matching can be case-insensitive; numbers must match exactly).
fields_to_check = ["applicant_name", "amount_ghs", "purpose", "monthly_profit_ghs",
                    "has_collateral_or_guarantor", "repayment_months"]

gold_letter_ids = list(GOLD.keys())  # ["L001", "L003", "L006"]

comparison_rows = []
for field in fields_to_check:
    row = {"field": field}
    matches = 0
    for letter_id in gold_letter_ids:
        gold_value = GOLD[letter_id][field]
        extracted_value = extraction_df.loc[letter_id, field]

        if field == "applicant_name":
            is_match = str(gold_value).strip().lower() == str(extracted_value).strip().lower()
        elif gold_value is None:
            is_match = pd.isna(extracted_value)
        else:
            is_match = gold_value == extracted_value

        row[letter_id] = "✓" if is_match else "✗"
        matches += int(is_match)

    row["accuracy"] = f"{matches}/{len(gold_letter_ids)}"
    comparison_rows.append(row)



# TODO: Display a small table: rows = fields, columns = L001 / L003 / L006 / accuracy.
accuracy_df = pd.DataFrame(comparison_rows).set_index("field")
accuracy_df

# %% [markdown]
# ### Part 4.2 — Reliability: is the system consistent?

# %%
# TODO: Run extract_fields() on letter L004 FIVE times at temperature=0 and FIVE times at
#   temperature=1.0.
def run_reliability_test(letter_id, temperature, n=5):
    raw_results = []
    valid_count = 0
    for i in range(n):
        result = extract_fields(LETTERS[letter_id], temperature=temperature)
        if result is not None:
            valid_count += 1
            raw_results.append(json.dumps(result, sort_keys=True))
        else:
            raw_results.append(None)
    unique_count = len(set(r for r in raw_results if r is not None))
    return valid_count, unique_count, raw_results


# TODO: For each temperature, report how many of the 5 runs produced (a) valid JSON and
#   (b) identical values across runs. A simple approach: json.dumps(result, sort_keys=True)
#   and count unique strings.
for temp in [0, 1.0]:
    valid, unique, raw_results = run_reliability_test("L004", temp, n=5)
    print(f" Temperature {temp} ")
    print(f"Valid JSON: {valid}/5")
    print(f"Identical result strings: {unique} unique out of 5 (1 = perfectly consistent)")
    for i, r in enumerate(raw_results, 1):
        print(f"  Run {i}: {r}")
    print()


# %% [markdown]
# ### Part 4.3 — Hallucination probing

# %%
# TODO: Design TWO adversarial tests and run them:
#   Test 1 — Ask your summarizer a question about a detail that is NOT in a letter
#     (e.g. "What is the applicant's credit score?"). Does it admit the information is
#     absent, or does it invent one?
#   Test 2 — Feed your extractor an EMPTY or IRRELEVANT text (e.g. a weather report).
#     Does it return nulls, or does it fabricate an applicant?
# Test 1: ask the summarizer about something not in the letter
test1_answer = ask_llm(
    "Based on this loan application, what is the applicant's credit score?\n\n" + LETTERS["L001"],
    system_prompt=SUMMARY_PROMPT_V2_SYSTEM,
    temperature=0,
)
print(" Test 1: asking for a detail not in the letter (credit score) ")
print(test1_answer)
print()

# Test 2: feed the extractor something irrelevant
weather_report = """Today's weather in Accra: sunny with a high of 31C, light winds from
the southwest, and a 10% chance of rain in the afternoon. Humidity around 65%."""

test2_result = extract_fields(weather_report)
print("Test 2: feeding the extractor an irrelevant text (weather report)")
print(test2_result)



# TODO: Record the outputs verbatim below and label each PASS or FAIL.
print("Adversarial Test Results\n")

print("Test 1 (summarizer asked about missing detail - credit score): PASS")
print("Output:", test1_answer)
print()

print("Test 2 (extractor fed an irrelevant weather report): PASS")
print("Output:", test2_result)

# %% [markdown]
# **Student Reasoning — Evaluation results**
# *1. Report your extraction accuracy. Which field was hardest for the model and why?*
# *2. What did the reliability experiment show about temperature and production systems?*
# *3. Did your system hallucinate under probing? If yes, how could the prompt (or the system
# design around it) reduce the risk?*
# 
# > **Answer:** [Double-click to edit]
# 1. Five of six fields scored perfectly (3/3): name, amount, profit, collateral, repayment months. Only purpose scored 0/3, but that's not a real failure, it's free text, and the model paraphrased correctly (for example, "buy a deep freezer and expand into frozen foods" vs GOLD's "buy deep freezer / expand into frozen foods"), just not word-for-word. Exact match works for fixed values, not for paraphrased text.
# 2. Running extraction on L004 five times at temperature 0 and five times at 1.0 gave identical results every time at both temperatures. Unlike Part 1.2's brainstorming test, extraction has one correct answer per field pulled straight from the text, so there's little room for temperature to introduce variation. Reliability here depends more on how constrained the task is than on temperature alone
# 3. No hallucination under probing. Asked about a credit score not in the letter, the summarizer said it wasn't provided instead of inventing one. Fed an irrelevant weather report, the extractor returned all nulls instead of fabricating an applicant. Both prompts explicitly instructed the model not to guess, showing that clear constraints genuinely reduce hallucination risk.

# %% [markdown]
# ### Part 4.4 — Appropriateness: should this system exist?
# No code in this part — just judgment, which is the scarcest skill in AI for business.

# %% [markdown]
# **Student Reasoning — Appropriateness**
# *1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions
# with your system, who could be unfairly harmed, and how? Consider applicants who write
# poorly in English but run solid businesses.*
# *2. Loan letters contain personal data. What are the implications of sending them to a
# third-party API in another country? What would you check before deploying this at a real
# Ghanaian microfinance institution?*
# *3. Name TWO concrete safeguards you would build around this system in production (think:
# human review points, logging, appeal processes, monitoring).*
# 
# > **Answer:** [Double-click to edit]
# 1. Applicants who write poorly in English but run solid businesses are most at risk. L002 (Kwame) got weaker treatment partly because his letter was vague and informal, not necessarily because his business was worse. A system judging text quality risks conflating writing ability with creditworthiness, disadvantaging less formally educated or non-native English speakers, people already underserved by traditional finance. Full automation removes the safety net the brief already builds in, inviting someone in to explain themselves.
# 
# 2. These letters contain real personal and financial data, sent to a server outside Ghana, outside local jurisdiction and the institution's direct control. Before real deployment, you'd need to check the provider's data retention and training policy, whether applicants consented to third-party sharing, compliance with Ghana's Data Protection Act, and whether a data processing agreement exists. Worth also checking if a local model or anonymized letters could avoid cross-border transfer entirely.
# 
# 3. Mandatory human review before any final decision, the system should never auto-approve or auto-reject, only suggest next steps. And full logging of every letter, output, and human decision, so outcomes can be audited for bias and applicants have something concrete to appeal against.

# %% [markdown]
# ---
# # Section 5 — Reflection
# 
# *Answer in a few sentences each:*
# 
# 1. **Prompting as engineering:** How is iterating on a prompt similar to and different from
#    iterating on the model hyperparameters you tuned in Lab 3?
# 2. **Trust:** After your Section 4 evaluation, would you trust this system to run unattended?
#    What single evaluation result most influenced your answer?
# 3. **Cost and scale:** Estimate (from your `response.usage` numbers) the tokens needed to
#    process 1,000 applications per month. What does that imply for provider choice?
# 4. **Looking back at the course:** You have now used classical ML (Lab 2), trained neural
#    networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one,
#    why does calling an API beat training your own model — and when would it not?
# 
# > **Answer:** [Double-click to edit]
# 
# 1. Both are trial-and-feedback loops, change something, test it, see the effect, adjust. The difference is what's being tuned. Hyperparameters are numeric and judged by a quantitative score, so the search can often be automated. Prompts are language, judged more qualitatively (factual, well-formatted, no hallucination), so intuition about wording matters more than math.
# 
# 2. Not for full automation, but yes as a decision-support layer with a human reviewing. What shaped this most was Part 3.3, where a strong application (L003) and a weak one (L006) got the identical next step, "invite for interview." That's a sign the system isn't risk-differentiating enough to trust without oversight.
# 
# 3. One application used 1,538 tokens across all three calls. At 1,000 applications a month, that's about 1,538,000 tokens, comfortably inside Groq's free tier for a small pilot, but cost would matter if usage grew or the provider changed.
# 
# 4. An API wins here because training a model needs data, compute, and expertise the institution doesn't have, while a foundation model already understands natural language, so prompting alone gets a working system fast. Training your own would make sense at large scale where API costs outgrow hosting your own model, or where data can never leave your own servers.

# %% [markdown]
# 

# %% [markdown]
# # Measuring real token usage across the full 3-call pipeline (summarize, extract, brief)
# # for one letter, to get an accurate cost estimate instead of relying on the single toy
# # question tested in Part 1.1.
# sample_letter = LETTERS["L001"]
# ...

# %%
sample_letter = LETTERS["L001"]

r1 = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SUMMARY_PROMPT_V2_SYSTEM},
        {"role": "user", "content": summaary_prompt_v2(sample_letter)},
    ],
    temperature=0,
)

r2 = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": extract_prompt(sample_letter)},
    ],
    temperature=0,
)

extracted_sample = extract_fields(sample_letter)
r3 = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": BRIEF_SYSTEM_PROMPT},
        {"role": "user", "content": brief_prompt(sample_letter, extracted_sample)},
    ],
    temperature=0,
)

print("Summarization tokens:", r1.usage.total_tokens)
print("Extraction tokens:", r2.usage.total_tokens)
print("Brief tokens:", r3.usage.total_tokens)

total_per_application = r1.usage.total_tokens + r2.usage.total_tokens + r3.usage.total_tokens
print("Total tokens for ONE application (3 calls):", total_per_application)

tokens_per_month = total_per_application * 1000
print(f"Estimated tokens for 1,000 applications/month: {tokens_per_month:,}")

# %% [markdown]
# ---
# ### Submission checklist
# 
# - [ ] All cells run top-to-bottom with no errors (`Kernel -> Restart & Run All`).
# - [ ] **No API key anywhere in the notebook or the commit history.**
# - [ ] Every **Student Reasoning** box is filled in with full sentences.
# - [ ] `prompts.py` / `prompts.md` committed with your final prompt templates.
# - [ ] Evaluation tables and adversarial test outputs visible in the saved notebook.
# - [ ] Notebook pushed to `lab-4-llm-decision-support` with incremental commits.
# - [ ] Repository link submitted to the course portal.
# - [ ] AI Declaration form in Repository.


