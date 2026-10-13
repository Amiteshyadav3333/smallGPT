"""
Indian politics ka training data banane ki script.
Run:  pip install requests
      python collect_data.py
Output: data.txt (clean text) + word count
"""
import os
import re
import time
import requests

LANG = "en"  # "hi" karoge to Hindi Wikipedia se aayega (topic names Hindi mein dene padenge)

TOPICS = [
    "Constitution of India", "Preamble to the Constitution of India",
    "Fundamental Rights in India", "Directive Principles of State Policy",
    "Fundamental Duties of Indian citizens", "President of India",
    "Vice President of India", "Prime Minister of India", "Council of Ministers",
    "Parliament of India", "Lok Sabha", "Rajya Sabha", "Speaker of the Lok Sabha",
    "Supreme Court of India", "High courts of India", "Judiciary of India",
    "Election Commission of India", "Elections in India",
    "1951–52 Indian general election", "Electronic voting machine",
    "Voter-verified paper audit trail", "Model Code of Conduct (India)",
    "Political parties in India", "Indian National Congress",
    "Bharatiya Janata Party", "Anti-defection law", "Governor (India)",
    "Chief minister (India)", "States and union territories of India",
    "Panchayati raj", "Local government in India", "Federalism in India",
    "Amendment of the Constitution of India", "Emergency (India)",
    "Indian independence movement", "Constituent Assembly of India",
    "B. R. Ambedkar", "Right to Information Act, 2005", "Goods and Services Tax (India)",
    "Comptroller and Auditor General of India", "Union Public Service Commission",
]

HEADERS = {"User-Agent": "learning-llm-project/1.0 (https://example.com; yadavamiteshkumar11@gmail.com)"}  # apna email daalo


def fetch(title):
    url = f"https://{LANG}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "redirects": 1,
    }
    for attempt in range(6):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:  # rate limit -> ruk ke dobara try
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"  429 mila, {wait}s ruk raha hoon...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        return next(iter(pages.values())).get("extract", "")
    raise RuntimeError("Rate limit se nahi nikla")


def cached_fetch(title):
    """Ek baar download hua topic raw/ folder mein save, dobara download nahi hoga."""
    os.makedirs("raw", exist_ok=True)
    path = os.path.join("raw", re.sub(r"[^A-Za-z0-9]+", "_", title) + ".txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    text = fetch(title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    time.sleep(2)  # har naye request ke beech gap
    return text


def clean(text):
    for a, b in {"\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}.items():
        text = text.replace(a, b)  # dash/quotes ko normal bana do, delete mat karo
    text = re.sub(r"==+[^=]+==+", "", text)          # section headings
    text = re.sub(r"\[\d+\]", "", text)               # [1] citations
    text = re.sub(r"[^\x00-\x7F\u0900-\u097F]+", " ", text)  # weird symbols (English + Hindi rakhta hai)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [l.strip() for l in text.split("\n")]
    return [l for l in lines if len(l.split()) >= 6]  # bahut chhoti lines hatao


# Verified facts -> Q&A format. Apne facts yahan aur jodo.
FACTS = [
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("When did the Constitution of India come into effect?", "The Constitution of India came into effect on 26 January 1950."),
    ("Who is the head of state in India?", "The President is the head of state of India."),
    ("Who is the head of government in India?", "The Prime Minister is the head of government of India."),
    ("How many elected seats are there in the Lok Sabha?", "The Lok Sabha has 543 elected seats."),
    ("What is the maximum strength of the Rajya Sabha?", "The Rajya Sabha has a maximum strength of 250 members."),
    ("What is the term of the Lok Sabha?", "The term of the Lok Sabha is five years unless it is dissolved earlier."),
    ("What is the term of a Rajya Sabha member?", "A Rajya Sabha member serves a term of six years."),
    ("Which body conducts elections in India?", "The Election Commission of India conducts elections in India."),
    ("What is the minimum voting age in India?", "The minimum voting age in India is 18 years."),
    ("When was the first general election of India held?", "The first general election of India was held in 1951-52."),
    ("Who chaired the drafting committee of the Constitution?", "B. R. Ambedkar chaired the drafting committee of the Constitution."),
]
STYLES = [
    ("Question: {q}\nAnswer: {a}\n"),
    ("Q: {q}\nA: {a}\n"),
    ("Question: {q}\nShort answer: {a}\n"),
]


def make_qa():
    out = []
    for q, a in FACTS:
        for s in STYLES:
            out.append(s.format(q=q, a=a))
    return out


def main():
    all_lines, seen = [], set()
    for t in TOPICS:
        try:
            print("Downloading:", t)
            for line in clean(cached_fetch(t)):
                if line not in seen:
                    seen.add(line)
                    all_lines.append(line)
        except Exception as e:
            print("  skip:", t, e)

    body = "\n".join(all_lines)
    qa = "\n".join(make_qa())
    final = body + "\n\n" + qa
    with open("data.txt", "w", encoding="utf-8") as f:
        f.write(final)

    words = len(final.split())
    print(f"\nDone. data.txt saved. Total words: {words}")
    if words < 100_000:
        print("Target 100k se kam hai -> TOPICS list badhao ya apne notes/Q&A jodo.")


if __name__ == "__main__":
    main()