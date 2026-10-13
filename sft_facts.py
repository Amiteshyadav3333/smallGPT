"""
sft_facts.py - SFT ke liye Q&A facts (pure Python, koi library nahi chahiye)

FACTS   : (key, answer, [5 alag tareeke ke sawal])
          Pehle 4 sawal training mein jate hain, 5th sawal TEST ke liye alag rakha jata hai.
          Isse pata chalta hai ki model ne naye shabdon wale sawal ka jawab diya ya nahi.
UNSEEN  : (key, answer, sawal) - ye fact training mein bilkul nahi jate. Honest test ke liye.
key     : wo shabd jo sahi answer mein hona chahiye (auto-check ke liye).

Apne facts jodo, par pehle verify karo - galat fact seedha model mein chala jayega.
Dhyan rahe: model character-level hai aur uski memory 128 characters ki hai, isliye
sawal + answer chhote rakho (sft.py lambe examples ko skip karke bata deta hai).
"""

FACTS = [
    ("president", "The President.", [
        "Who is the head of state in India?",
        "Which office is the head of state of India?",
        "Name the head of state of India.",
        "India's head of state holds which office?",
        "Who is the head of the Indian state?"]),
    ("prime minister", "The Prime Minister.", [
        "Who is the head of government in India?",
        "Who leads the government of India?",
        "Which office heads the Union Council of Ministers?",
        "Name the head of government of India.",
        "Who is the head of the Indian government?"]),
    ("president", "The President.", [
        "Who has the power to appoint the Prime Minister?",
        "By whom is the Prime Minister of India appointed?",
        "Which authority appoints the Prime Minister of India?",
        "The Prime Minister of India is appointed by whom?",
        "Who appoints the Prime Minister of India?"]),
    ("26 january 1950", "26 January 1950.", [
        "When did the Constitution of India come into effect?",
        "On what date did the Indian Constitution come into force?",
        "When was the Constitution of India enforced?",
        "The Constitution of India became effective on which date?",
        "From which date is the Indian Constitution in force?"]),
    ("26 november 1949", "26 November 1949.", [
        "When was the Constitution of India adopted?",
        "On what date did the Constituent Assembly adopt the Constitution?",
        "When did the Constituent Assembly adopt the Indian Constitution?",
        "What is the adoption date of the Indian Constitution?",
        "The Indian Constitution was adopted on which date?"]),
    ("543", "543 seats.", [
        "How many elected seats are there in the Lok Sabha?",
        "What is the number of elected members in the Lok Sabha?",
        "How many members are elected to the Lok Sabha?",
        "Lok Sabha has how many elected seats?",
        "How many constituencies elect members to the Lok Sabha?"]),
    ("five years", "Five years.", [
        "What is the term of the Lok Sabha?",
        "For how many years is the Lok Sabha elected?",
        "How long does the Lok Sabha last?",
        "What is the tenure of the Lok Sabha?",
        "What is the normal duration of a Lok Sabha?"]),
    ("250", "250 members.", [
        "What is the maximum strength of the Rajya Sabha?",
        "How many members can the Rajya Sabha have at most?",
        "What is the upper limit of Rajya Sabha membership?",
        "What is the largest number of members in the Rajya Sabha?",
        "The Rajya Sabha can have how many members at maximum?"]),
    ("six years", "Six years.", [
        "What is the term of a Rajya Sabha member?",
        "For how many years does a Rajya Sabha member hold office?",
        "How long is the tenure of a Rajya Sabha member?",
        "A member of the Rajya Sabha is elected for how many years?",
        "What is the length of a Rajya Sabha member's term?"]),
    ("rajya sabha", "The Rajya Sabha.", [
        "Which house of Parliament is never dissolved?",
        "Which house of Parliament is a permanent house?",
        "Which house of Parliament cannot be dissolved?",
        "Name the permanent house of Parliament.",
        "Which house of Parliament is always in existence?"]),
    ("speaker", "The Speaker.", [
        "Who presides over the Lok Sabha?",
        "Who is the presiding officer of the Lok Sabha?",
        "Who chairs the meetings of the Lok Sabha?",
        "Name the presiding officer of the Lok Sabha.",
        "Which officer conducts the proceedings of the Lok Sabha?"]),
    ("election commission", "The Election Commission of India.", [
        "Which body conducts elections in India?",
        "Who conducts elections in India?",
        "Which institution is responsible for holding elections in India?",
        "Name the body that conducts elections in India.",
        "Which authority organises elections in India?"]),
    ("18", "18 years.", [
        "What is the minimum voting age in India?",
        "At what age can a person vote in India?",
        "From what age can an Indian citizen vote?",
        "What is the legal voting age in India?",
        "How old must a citizen be to vote in India?"]),
    ("35", "35 years.", [
        "What is the minimum age to become the President of India?",
        "How old must a person be to become the President of India?",
        "What is the minimum age required for the President of India?",
        "At what minimum age can one contest for President?",
        "What age is needed to be the President of India?"]),
    ("25", "25 years.", [
        "What is the minimum age to contest the Lok Sabha election?",
        "How old must a candidate be to contest the Lok Sabha election?",
        "What is the minimum age to become a Lok Sabha member?",
        "At what minimum age can one stand for the Lok Sabha?",
        "What age is required to be a member of the Lok Sabha?"]),
    ("30", "30 years.", [
        "What is the minimum age to become a Rajya Sabha member?",
        "How old must a person be to be a member of the Rajya Sabha?",
        "What is the minimum age required for the Rajya Sabha?",
        "At what minimum age can one become a Rajya Sabha member?",
        "What age is needed to sit in the Rajya Sabha?"]),
    ("five years", "Five years.", [
        "What is the term of the President of India?",
        "For how many years is the President of India elected?",
        "How long does the President of India hold office?",
        "What is the tenure of the President of India?",
        "The President of India serves for how many years?"]),
    ("1951", "1951-52.", [
        "When was the first general election of India held?",
        "In which year was India's first general election held?",
        "When did India hold its first general election?",
        "India's first general election took place in which year?",
        "In which period was the first Lok Sabha election conducted?"]),
    ("ambedkar", "B. R. Ambedkar.", [
        "Who chaired the drafting committee of the Constitution?",
        "Who was the chairman of the Drafting Committee?",
        "Who headed the drafting committee of the Indian Constitution?",
        "Name the chairman of the drafting committee of the Constitution.",
        "Which leader chaired the Constitution drafting committee?"]),
    ("rajendra prasad", "Rajendra Prasad.", [
        "Who was the president of the Constituent Assembly?",
        "Who presided over the Constituent Assembly of India?",
        "Who was the chairman of the Constituent Assembly?",
        "Name the president of the Constituent Assembly.",
        "Who headed the Constituent Assembly?"]),
    ("rajendra prasad", "Rajendra Prasad.", [
        "Who was the first President of India?",
        "Who became the first President of the Republic of India?",
        "Name the first President of India.",
        "Who was India's first President?",
        "Who held the office of President of India first?"]),
    ("nehru", "Jawaharlal Nehru.", [
        "Who was the first Prime Minister of India?",
        "Who became the first Prime Minister of independent India?",
        "Name the first Prime Minister of India.",
        "Who was India's first Prime Minister?",
        "Who held the office of Prime Minister of India first?"]),
    ("supreme court", "The Supreme Court.", [
        "Which is the highest court in India?",
        "What is the highest court of India?",
        "Which court is at the top of the Indian judiciary?",
        "Name the highest court in India.",
        "Which is the apex court of India?"]),
    ("president", "The President.", [
        "Who appoints the Chief Justice of India?",
        "By whom is the Chief Justice of India appointed?",
        "Which authority appoints the Chief Justice of India?",
        "The Chief Justice of India is appointed by whom?",
        "Who names the Chief Justice of India?"]),
    ("six", "Six.", [
        "How many fundamental rights are there in the Constitution?",
        "What is the number of fundamental rights in India?",
        "How many fundamental rights does an Indian citizen have?",
        "The Constitution guarantees how many fundamental rights?",
        "How many fundamental rights exist at present in India?"]),
    ("42nd", "The 42nd Amendment.", [
        "Which amendment added the Fundamental Duties?",
        "By which amendment were the Fundamental Duties added?",
        "Fundamental Duties were added by which Amendment?",
        "Which Amendment introduced Fundamental Duties?",
        "Through which amendment did Fundamental Duties enter the Constitution?"]),
    ("part iii", "Part III.", [
        "In which part of the Constitution are the Fundamental Rights?",
        "Which part of the Indian Constitution contains Fundamental Rights?",
        "Fundamental Rights are given in which part?",
        "Where are Fundamental Rights placed in the Constitution?",
        "Which part of the Constitution deals with Fundamental Rights?"]),
    ("part iv", "Part IV.", [
        "In which part of the Constitution are the Directive Principles?",
        "Which part of the Constitution contains the Directive Principles?",
        "The Directive Principles of State Policy are in which part?",
        "Where are the Directive Principles placed in the Constitution?",
        "Which part of the Constitution deals with Directive Principles?"]),
    ("sovereign", "A sovereign, socialist, secular, democratic republic.", [
        "How does the Preamble describe India?",
        "What does the Preamble call India?",
        "According to the Preamble, what kind of state is India?",
        "Which words in the Preamble describe the nature of India?",
        "What is India called in the Preamble of the Constitution?"]),
    ("life", "Right to life and personal liberty.", [
        "What does Article 21 deal with?",
        "What right is protected by Article 21?",
        "What is Article 21 of the Constitution about?",
        "Article 21 of the Indian Constitution is about which right?",
        "What does Article 21 protect?"]),
    ("equality", "Equality before the law.", [
        "What does Article 14 deal with?",
        "What right is guaranteed by Article 14?",
        "What is Article 14 of the Constitution about?",
        "Article 14 of the Indian Constitution is about which right?",
        "What does Article 14 guarantee?"]),
    ("amendment", "Amendment of the Constitution.", [
        "What does Article 368 deal with?",
        "What is Article 368 of the Constitution about?",
        "What power is given by Article 368?",
        "Article 368 of the Indian Constitution is about what?",
        "What is the subject of Article 368?"]),
    ("president's rule", "President's Rule in a state.", [
        "What does Article 356 deal with?",
        "What is Article 356 of the Constitution about?",
        "What is provided for by Article 356?",
        "Article 356 of the Indian Constitution is about what?",
        "What is the subject of Article 356?"]),
    ("73rd", "The 73rd Amendment.", [
        "Which amendment gave constitutional status to Panchayati Raj?",
        "Which amendment made Panchayati Raj constitutional bodies?",
        "Panchayati Raj got constitutional status through which amendment?",
        "Which Amendment is related to Panchayati Raj?",
        "What gave Panchayati Raj institutions constitutional status?"]),
    ("74th", "The 74th Amendment.", [
        "Which amendment gave constitutional status to urban local bodies?",
        "Which amendment made municipalities constitutional bodies?",
        "Urban local bodies got constitutional status through which amendment?",
        "Which Amendment is related to municipalities?",
        "What gave urban local bodies constitutional status?"]),
    ("president", "The President.", [
        "Who appoints the Governor of a state?",
        "By whom is the Governor of a state appointed?",
        "Which authority appoints the Governor of a state?",
        "The Governor of a state is appointed by whom?",
        "Who names the Governor of an Indian state?"]),
    ("governor", "The Governor.", [
        "Who appoints the Chief Minister of a state?",
        "By whom is the Chief Minister of a state appointed?",
        "Which authority appoints the Chief Minister of a state?",
        "The Chief Minister of a state is appointed by whom?",
        "Who names the Chief Minister of an Indian state?"]),
    ("28", "28 states and 8 union territories.", [
        "How many states does India have?",
        "What is the number of states in India?",
        "How many states and union territories does India have?",
        "India is divided into how many states?",
        "What is the total number of states in India?"]),
    ("2004", "2004.", [
        "When were EVMs first used across all of India?",
        "In which general election were EVMs used in the whole country?",
        "EVMs were used all over India for the first time in which year?",
        "In which year did India vote fully on EVMs for the first time?",
        "When did EVMs replace ballot papers across the country?"]),
    ("paper slip", "It prints a paper slip to verify the vote.", [
        "What is VVPAT?",
        "What does a VVPAT machine do?",
        "What is the purpose of VVPAT?",
        "Explain VVPAT in one line.",
        "What is a voter verified paper audit trail?"]),
    ("none of the above", "None of the Above.", [
        "What does NOTA stand for?",
        "What is the full form of NOTA?",
        "Expand the term NOTA.",
        "NOTA in elections means what?",
        "What is NOTA in an election?"]),
    ("2005", "2005.", [
        "When was the Right to Information Act passed?",
        "In which year was the RTI Act passed?",
        "The Right to Information Act came in which year?",
        "In which year did India enact the RTI Act?",
        "What is the year of the RTI Act?"]),
    ("2017", "1 July 2017.", [
        "When was GST introduced in India?",
        "On which date did GST come into effect in India?",
        "In which year was GST implemented in India?",
        "When did India start the Goods and Services Tax?",
        "GST was launched in India on which date?"]),
    ("audits", "It audits Union and state accounts.", [
        "What does the Comptroller and Auditor General do?",
        "What is the function of the CAG?",
        "What is the role of the CAG of India?",
        "What work does the Comptroller and Auditor General perform?",
        "What is the duty of the CAG?"]),
    ("union public service commission", "The Union Public Service Commission.", [
        "Which body conducts the civil services examination?",
        "Who conducts the civil services exam in India?",
        "Which commission holds the IAS examination?",
        "Name the body that conducts the civil services exam.",
        "Which institution recruits civil servants through exams?"]),
    ("tenth schedule", "The Tenth Schedule.", [
        "Which schedule contains the anti-defection law?",
        "Which Schedule of the Constitution deals with anti-defection?",
        "The anti-defection law is in which Schedule?",
        "Where is the anti-defection law placed in the Constitution?",
        "Which Schedule is related to defection of legislators?"]),
    ("lok sabha", "The Lok Sabha.", [
        "In which house can a Money Bill be introduced?",
        "Where can a Money Bill be introduced?",
        "In which house of Parliament is a Money Bill introduced?",
        "Which house can introduce a Money Bill?",
        "A Money Bill originates in which house?"]),
    ("new delhi", "New Delhi.", [
        "What is the capital of India?",
        "Which city is the capital of India?",
        "Name the capital of India.",
        "India's capital city is which city?",
        "Where is the capital of India located?"]),
    ("directly", "Directly by the people.", [
        "How are the members of the Lok Sabha elected?",
        "Who elects the members of the Lok Sabha?",
        "By whom are Lok Sabha members elected?",
        "In what manner are Lok Sabha members chosen?",
        "Lok Sabha members are chosen by whom?"]),
    ("65", "65 years.", [
        "At what age do Supreme Court judges retire?",
        "What is the retirement age of a Supreme Court judge?",
        "Supreme Court judges retire at what age?",
        "Up to what age can a Supreme Court judge serve?",
        "What is the age of retirement for Supreme Court judges?"]),
    ("chief election commissioner", "The Chief Election Commissioner.", [
        "Who heads the Election Commission of India?",
        "Which officer is the head of the Election Commission?",
        "Name the head of the Election Commission of India.",
        "The Election Commission is headed by whom?",
        "Who is the chairperson of the Election Commission?"]),
    ("vice president", "The Vice President of India.", [
        "Who is the chairman of the Rajya Sabha?",
        "Who presides over the Rajya Sabha?",
        "Who is the presiding officer of the Rajya Sabha?",
        "Name the ex officio chairman of the Rajya Sabha.",
        "Which office holder chairs the Rajya Sabha?"]),
]

# Ye facts training mein NAHI jate. Model ne inke jawab pretraining text se ya kahin se
# seekhe honge to sahi aayenge, warna galat - ye knowledge ka asli test hai.
UNSEEN = [
    ("12", "12 members.", "How many members does the President nominate to the Rajya Sabha?"),
    ("22", "22 languages.", "How many languages are listed in the Eighth Schedule?"),
    ("62", "62 years.", "At what age do High Court judges retire?"),
    ("speaker", "The Speaker.", "Who presides over a joint sitting of both Houses of Parliament?"),
    ("assemblies", "Elected members of state legislative assemblies.", "Who elects the members of the Rajya Sabha?"),
    ("president", "The President.", "Who can dissolve the Lok Sabha?"),
    ("32", "Article 32.", "Which article gives the right to constitutional remedies?"),
    ("five years", "Five years.", "What is the normal term of a Governor?"),
]