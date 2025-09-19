#!/usr/bin/env python3
"""
Generate Finnish clinical notes from guideline metadata (title + keywords) and save to JSONL:

Input files (same schema for both):
  {
    "guideline_id": "hoi_04010",
    "title": "Kohonnut verenpaine",
    "keywords": ["Sisätaudit", "Kardiologia", ...],
    "chunks": [ {"chunk_id": "...", "page_content": "..."}, ... ]
  }

Output JSONL (clinical_notes_gpt.jsonl):
  {
    "note_id": "note_0001",
    "text": "...clinical note text...",
    "linked_guideline_ids": ["hoi_04010"]
  }

Usage:
  export OPENAI_API_KEY=sk-...
  python make_clinical_notes_from_guidelines.py \
    --kaypa ./json_guidelines/Käypä_hoito.jsonl \
    --valta ./json_guidelines/Vältä_viisaasti.jsonl \
    --out ./clinical_notes_gpt.jsonl \
    --notes-per-guideline 10 \
    --model gpt-4o-mini

Notes:
- Uses OpenAI's modern Python SDK (`pip install openai>=1.0.0`).
- Deterministic-ish variety via a pool of symptom seeds + temperature.
- Ensures required heading order and terse style.
- By default links only the current guideline_id. (Optional RRF labeling could be added later.)
"""

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Any

# --- config ---
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/") 
from utils import read_secrets, load_paths, load_jsonl

secrets = read_secrets()
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
OPENAI_MODEL = secrets["OPENAI_MODEL"]

paths = load_paths()
GUIDELINE_JSON_DIR = paths["GUIDELINE_JSON_DIR"]
GUIDELINE_FAISS_DIR = paths["GUIDELINE_FAISS_DIR"]
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]


# --- LLM client (OpenAI modern SDK) ---
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None


PROMPT_BASE = (
    "Olet lääkäri suomalaisessa perusterveydenhuollossa (kiireetön vastaanotto terveysasemalla). \
    Laadi ytimekäs, telegrafinen lääkärimerkintä suomeksi lääkärien vakiintuneella kielellä käyttäen tavanomaisia lyhenteitä. \
    Käytä täsmälleen tätä otsikkojärjestystä: Tulosyy, Esitiedot, Nykytila, Suunnitelma, Diagnoosit. \
    Yleiset ohjeet: vain olennainen tieto, ei selittelyä, tiiviisti. Päivämäärät lyhyesti, esim. 'viim pe 23.4.'. \
    Kielteiset löydökset lyhyesti, esim. 'ei kuumetta, ei hengenahdistusta'. Älä käytä täytesanoja. \
    Älä lisää muita otsikoita tai loppukaneettia. Esitiedot: yksi kappale. Potilas on X ikäinen nainen/mies, \
    Tulosyy: esim. 'kuume ja kurkkukipu'/'Virtsaamiskipu'. Olennaiset taustasairaudet ja riskitekijät, \
    säännöllinen lääkitys ja lääkeallergiat, oireiden kulku ja aiemmat hoidot. Nykytila: vain ytimekkäät löydökset. \
    Vitaalit vain olennaiset muodossa: RR xx/xx, syke xx/min, rf xx/min, temp xx.x C, SpO2 xx %. \
    Status vain asiaankuuluvalta osin. Suunnitelma: yksi kappale, enintään 2-3 virkettä. 1) Tutkimukset ja kiireellisyys lyhyesti. \
    2) Hoito ja uusi lääkitys annoksineen yhdessä virkkeessä. 3) Lyhyet yhteydenottorajat. Ei listoja, ei laajoja kotihoito-ohjeita, \
    ei pitkää varoitusluetteloa. Sairausloma muodossa 'SA x pv', lausunnot tai lähetteet vain jos tarpeen. \
    Diagnoosit: listaa ensisijaisuusjärjestyksessä muodossa ICD-10-koodi + nimi."
)

def gen_tulosyy_rules(title: str, keywords: list) -> str:
    txt = (title or "") + " " + " ".join(k for k in (keywords or []) if isinstance(k, str))
    txt = txt.lower()
    def pick(xs):
        return random.choice(xs)
    if any(w in txt for w in ["ekseema","dermati","atoop","ihottu","kutina","allerg","prur"]):
        return pick(["kutina","ihottuma pahentunut","yöllinen kutina","raapimisjäljet ja kutina"]) 
    if any(w in txt for w in ["nielu","kurkku","angiina","tonsil","strepto"]):
        return pick(["kuume ja kurkkukipu","nielemiskipu","kurkun turvotus"]) 
    if any(w in txt for w in ["virtsa","kysti","uretri","pyel"]):
        return pick(["virtsaamiskipu ja kirvely","tiheävirtsaisuus","verivirtsaisuus"]) 
    if any(w in txt for w in ["korva","otitis","otit"]):
        return pick(["korvakipu","korvakipu ja kuume","korvan tukkoisuus"]) 
    if any(w in txt for w in ["yskä","keuhko","bronki","astma","copd"]):
        return pick(["pitkittynyt yskä","yskä ja hengenahdistus","hengityksen vinkuna"]) 
    if any(w in txt for w in ["rinta","angina","sydän","verenpaine","hypertens"]):
        return pick(["rintakipu","kohonnut verenpaine","huimaus ja rintakipu"]) 
    if any(w in txt for w in ["päänsärky","mig","neurol","aiv"]):
        return pick(["päänsärky","päänsärky ja pahoinvointi"]) 
    if any(w in txt for w in ["gastro","vatsa","ripul","oksenn"]):
        return pick(["vatsakipu","vatsakipu ja ripuli","pahoinvointi ja oksentelu"]) 
    return pick(["yleinen tarkastuskäynti","kuume","väsymys"]) 


def gen_tulosyy_llm(client, model: str, title: str, keywords: list, seed: int = 0) -> list:
    """Return a small list of Tulosyy candidates for this guideline (1-4 words each)."""
    sys_msg = (
        "Laadi 6 lyhyttä Tulosyy-vaihtoehtoa suomeksi perusterveydenhuollon vastaanotolle, \
        sovita aihe otsikkoon ja avainsanoihin. Jokainen 1-4 sanaa, ei pisteitä, ei kysymysmerkkiä, \
        ei diagnoosin nimiä (vain potilaan ilmoittama päävaiva). Palauta JSON-taulukkona (merkkijonoja)."
    )
    user_msg = f"Otsikko: {title}, Avainsanat: {', '.join(k for k in (keywords or []) if isinstance(k, str))}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
            temperature=0.6
        )
        content = resp.choices[0].message.content.strip()
        arr = []
        try:
            arr = json.loads(content)
        except Exception:
            # best-effort: split lines
            arr = [x.strip("- •* ") for x in content.splitlines() if x.strip()]
        arr = [s for s in arr if isinstance(s, str) and 1 <= len(s.split()) <= 4]
        # de-dup while preserving order
        seen = set(); dedup = []
        for s in arr:
            t = s.strip()
            if t and t not in seen:
                seen.add(t); dedup.append(t)
        if dedup:
            return dedup[:6]
    except Exception:
        pass
    # fallback to minimal rules if LLM fails
    return [gen_tulosyy_rules(title, keywords)]


def build_instruction(PROMPT_BASE: str, title: str, keywords: List[str]) -> str:
    kw = ", ".join([k for k in keywords if isinstance(k, str)]) if keywords else ""
    ctx = (
        f"Kliininen teema ohjataan seuraavien tietojen mukaan. Otsikko: '{title}'. Avainsanat: [{kw}]. "
        f"Luo uskottava, potilasturvallinen terveyskeskuksen merkintä, jonka pääaihe ja valinnat (status/diagnostiikka/hoito) ovat "
        f"luontevasti linjassa otsikon ja avainsanojen kanssa. Vaihtele esitietojen yksityiskohtia, löydöksiä ja suunnitelmaa, "
        f"mutta pidä rakenne ja tiiviysvaatimukset täsmälleen ohjeen mukaisina. Älä viittaa suosituksen nimeen tai lähteeseen tekstissä."
    )
    return PROMPT_BASE + ctx


def build_user_prompt(age_sex: str, tulosyy: str) -> str:
    return (
        f"Käytä potilasesimerkkiä: {age_sex}. Aseta Tulosyy: {tulosyy}. "
        f"Muista lyhenteet ja mittayksiköt täsmälleen pyydetyssä muodossa."
    )


def gen_notes_for_guideline(client, model: str, title: str, keywords: List[str], n: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    notes: List[str] = []
    print("keywords import to gen_notes_for_guideline: ", keywords)
    sys_prom = build_instruction(PROMPT_BASE, title, keywords)
    
    def random_age_sex():
        age = random.randint(10, 80)  # random age
        sex = random.choice(["nainen", "mies"])  # random sex
        return f"{age}-v {sex}"

    # Generate Tulosyy candidates ONCE per guideline
    tulosyy_candidates = gen_tulosyy_llm(client, model, title, keywords, seed=seed)
    if not tulosyy_candidates:
        tulosyy_candidates = [gen_tulosyy_rules(title, keywords)]

    for i in range(n):
        age_sex = random_age_sex()
        tulosyy = random.choice(tulosyy_candidates)
        user_prom = build_user_prompt(age_sex, tulosyy)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prom},{"role":"user","content":user_prom}],
            temperature=0.8
        )
        note = resp.choices[0].message.content.strip()
        notes.append(ensure_format(note))
    return notes[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaypa", default="/scratch/work/zhangl9/genaid/TestBed/datasets/json_guidelines/Käypä_hoito.jsonl", help="Path to Käypä_hoito.jsonl")
    ap.add_argument("--valta", default="/scratch/work/zhangl9/genaid/TestBed/datasets/json_guidelines/Vältä_viisaasti.jsonl", help="Path to Vältä_viisaasti.jsonl")
    ap.add_argument("--out", default="./json_clinical_notes/clinical_notes_gpt.jsonl")
    ap.add_argument("--notes-per-guideline", type=int, default=10)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if OpenAI is None:
        print("[ERROR] openai SDK not installed. Run: pip install openai>=1.0.0")
        sys.exit(1)
    if not OPENAI_API_KEY:
        print("[ERROR] Please set OPENAI_API_KEY env var.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    def iter_guidelines(path: str):
        items = load_jsonl(path)
        for it in items:
            gid = it.get("guideline_id")
            title = it.get("title", "")
            keywords = it.get("keywords", [])
            if gid and title is not None:
                yield gid, title, keywords

    # all_guidelines = list(iter_guidelines(args.kaypa_hoito if hasattr(args, 'kaypa_hoito') else args.kaypa_hoito))
    # argparse converts --kaypa-hoito to kaypa_hoito; fix attribute names
    # if not all_guidelines:
        # try proper attr names
        # all_guidelines = list(iter_guidelines(getattr(args, 'kaypa_hoito', args.__dict__.get('kaypa-hoito'))))
    kh_guidelines = list(iter_guidelines(args.kaypa))
    vv_guidelines = list(iter_guidelines(args.valta))

    guidelines = kh_guidelines + vv_guidelines

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    note_idx = 1
    with open(args.out, 'w', encoding='utf-8') as f:
        for gid, title, keywords in guidelines:
            print("gid: ", gid)
            print("title: ", title)
            print("keywords: ", keywords)
            print("-"*100)
            notes = gen_notes_for_guideline(client, args.model, title, keywords, args.notes_per_guideline, seed=args.seed + note_idx)
            for note in notes:
                row = {
                    "note_id": f"note_{note_idx:04d}",
                    "text": note,
                    "linked_guideline_ids": [gid]
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                note_idx += 1
                print("note_idx: ", f"note_{note_idx:04d}")
                print("text: ", note)
                print("linked_guideline_ids: ", gid)
                print("-"*100)


    print(f"Wrote {note_idx - 1} notes -> {args.out}")


if __name__ == "__main__":
    main()
