"""Structured clinical extraction and standardisation.

Pipeline
--------
1. **Clause segmentation** – the note is split into clauses on sentence
   punctuation, ``;``, ``|`` and new lines (decimal points are preserved).
2. **Concept matching** – each canonical diagnosis / medication has a lexicon of
   surface forms: full terms (case-insensitive), abbreviations (case-sensitive,
   so ``AF`` does not fire on ``af`` inside words), brand names and spelling
   variants (``Lasix``/``frusemide`` -> ``furosemide``, ``Eliquis`` -> ``apixaban``).
3. **Context filtering (NegEx-style)** – a concept is dropped when its clause
   contains a pre-negation trigger before it (``denies``, ``no evidence of``),
   a post-trigger after it (``considered but not confirmed``, ``not started``,
   ``discontinued``), a family-history cue, or (for drugs) an allergy cue.
4. **Numeric parsing** – cue + number + optional unit, decimal commas accepted,
   units converted to the canonical unit (µmol/L -> mg/dL, g/L -> g/dL,
   mmol/L Hb -> g/dL) and range-checked. Explicit "not documented" statements
   and absent values both yield ``None``.
5. **Categorical fields** – ordered pattern tables for smoking status and the
   allergy class, evaluated inside the relevant clause(s).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Lexicons. Each entry: (pattern, case_sensitive)
# ---------------------------------------------------------------------------
_I = False  # case-insensitive
_S = True   # case-sensitive (abbreviations)

DIAGNOSIS_LEXICON: dict[str, list[tuple[str, bool]]] = {
    "atrial_fibrillation": [
        (r"atrial fibrillation", _I), (r"a-?fib", _I), (r"vorhofflimmern", _I),
        (r"AF", _S), (r"AFib", _S), (r"PAF", _S),
    ],
    "heart_failure": [
        (r"heart failure", _I), (r"cardiac failure", _I), (r"LV failure", _I),
        (r"left ventricular failure", _I), (r"cardiac insufficiency", _I), (r"herzinsuffizienz", _I),
        (r"HFrEF", _S), (r"HFpEF", _S), (r"HFmrEF", _S), (r"CHF", _S), (r"CCF", _S), (r"LVF", _S),
    ],
    "hypertension": [
        (r"hypertension", _I), (r"hypertensive (?:disease|heart disease)", _I),
        (r"high blood pressure", _I), (r"(?:arterielle )?hypertonie", _I), (r"HTN", _S),
    ],
    "type_2_diabetes": [
        (r"type (?:2|ii) (?:diabetes(?: mellitus)?|dm)", _I), (r"diabetes mellitus(?: type (?:2|ii))?", _I),
        (r"type 2 dm", _I), (r"diabetes type (?:2|ii)", _I), (r"non-insulin[- ]dependent diabetes", _I),
        (r"T2DM", _S), (r"DM2", _S), (r"DM ?II", _S), (r"T2D", _S), (r"NIDDM", _S), (r"DM", _S),
    ],
    "chronic_kidney_disease": [
        (r"chronic kidney disease", _I), (r"chronic renal (?:disease|dysfunction|insufficiency|failure|impairment)", _I),
        (r"chronische niereninsuffizienz", _I), (r"CKD", _S),
    ],
    "coronary_artery_disease": [
        (r"coronary (?:artery |heart )?disease", _I), (r"isch(?:a)?emic heart disease", _I),
        (r"koronare herzkrankheit", _I), (r"CAD", _S), (r"IHD", _S), (r"CHD", _S), (r"KHK", _S),
    ],
    "acute_coronary_syndrome": [
        (r"acute coronary syndrome", _I), (r"unstable angina", _I), (r"(?:non-?ST|ST)[- ]elevation myocardial infarction", _I),
        (r"acute myocardial infarction", _I), (r"ACS", _S), (r"N?STEMI", _S),
    ],
    "pneumonia": [
        (r"pneumonia", _I), (r"bronchopneumonia", _I), (r"infective consolidation", _I),
        (r"pneumonie", _I), (r"CAP", _S),
    ],
    "copd": [
        (r"chronic obstructive (?:pulmonary|airways?|lung) disease", _I), (r"chronic airways? obstruction", _I),
        (r"obstructive (?:lung|airways?) disease", _I), (r"COPD", _S), (r"COAD", _S),
    ],
}

MEDICATION_LEXICON: dict[str, list[tuple[str, bool]]] = {
    "apixaban": [(r"apixaban", _I), (r"eliquis", _I), (r"APX", _S)],
    "rivaroxaban": [(r"rivaroxaban", _I), (r"xarelto", _I)],
    "warfarin": [(r"warfarin", _I), (r"coumadin", _I), (r"marevan", _I)],
    "metoprolol": [(r"metoprolol", _I), (r"lopressor", _I), (r"betaloc", _I), (r"beloc(?:-zok)?", _I), (r"toprol", _I)],
    "bisoprolol": [(r"bisoprolol", _I), (r"concor", _I)],
    "furosemide": [(r"furosemide", _I), (r"frusemide", _I), (r"lasix", _I)],
    "ramipril": [(r"ramipril", _I), (r"tritace", _I), (r"altace", _I), (r"delix", _I)],
    "amlodipine": [(r"amlodipine?", _I), (r"norvasc", _I), (r"amlong", _I)],
    "metformin": [(r"metformin", _I), (r"glucophage", _I), (r"glycomet", _I)],
    "insulin": [(r"insulin(?: glargine| detemir| aspart| lispro)?", _I), (r"lantus", _I), (r"levemir", _I), (r"novorapid", _I)],
    "atorvastatin": [(r"atorvastatin", _I), (r"atorva", _I), (r"lipitor", _I), (r"sortis", _I)],
    "aspirin": [(r"aspirin", _I), (r"acetylsalicylic acid", _I), (r"ecosprin", _I), (r"ASA", _S), (r"ASS", _S)],
    "clopidogrel": [(r"clopidogrel", _I), (r"plavix", _I), (r"iscover", _I)],
    "amiodarone": [(r"amiodarone?", _I), (r"cordarone", _I)],
    "digoxin": [(r"digoxin", _I), (r"lanoxin", _I)],
    "azithromycin": [(r"azithromycin", _I), (r"azithro", _I), (r"zithromax", _I), (r"azee", _I)],
}

# Type-1 diabetes must never be mapped to type_2_diabetes.
_TYPE1_DIABETES = re.compile(r"(?i:type (?:1|i) (?:diabetes|dm)|\bT1DM\b|\bIDDM\b|juvenile diabetes)")

# ---------------------------------------------------------------------------
# Context triggers (NegEx-inspired). Evaluated inside a single clause.
# ---------------------------------------------------------------------------
_PRE_NEGATION = re.compile(
    r"(?i:\b(?:no|not|denies|denied|deny|without|negative for|free of|absence of|absent|ruled out|rule out|"
    r"r/o|excluded|no evidence of|no history of|no signs? of|no clinical features of|never had|kein(?:e|en)?|"
    r"not on|off|avoid(?:ed)?|contraindicated|suspected|possible|query|questionable)\b)|(?:^|\s)\?\s*$"
)
_POST_NEGATION = re.compile(
    r"(?i:\b(?:considered but|not confirmed|ruled out|was excluded|excluded|unlikely|not started|"
    r"was not started|not commenced|discussed but|discontinued|stopped|ceased|held|withheld|on hold|"
    r"declined|refused|not tolerated|was considered|to be considered|planned|to start|if needed)\b)"
)
_FAMILY = re.compile(
    r"(?i:\b(?:family history|fam(?:ily)? hx|FHx?|mother|father|brother|sister|sibling|parent|grandmother|"
    r"grandfather|aunt|uncle|cousin|relative|degree relative)\b)"
)
_ALLERGY_CONTEXT = re.compile(
    r"(?i:allerg|intoleran|hypersensitiv|anaphyla|urticaria|rash|reaction|nkda|unvertr)"
)
_CONJUNCTION = re.compile(r"(?i:\b(?:but|however|although|except|apart from)\b)")

_CLAUSE_SPLIT = re.compile(r"(?:(?<=[.!?])\s+(?=[A-Z\[(])|[;\n|]|(?<=\])\s|\.\s*$)")


@dataclass(frozen=True)
class Clause:
    start: int
    text: str


def split_clauses(note: str) -> list[Clause]:
    clauses: list[Clause] = []
    cursor = 0
    for match in _CLAUSE_SPLIT.finditer(note):
        if match.start() > cursor:
            clauses.append(Clause(cursor, note[cursor : match.start()]))
        cursor = match.end()
    if cursor < len(note):
        clauses.append(Clause(cursor, note[cursor:]))
    return [c for c in clauses if c.text.strip()]


def _compile_lexicon(lexicon: dict[str, list[tuple[str, bool]]]) -> dict[str, list[re.Pattern[str]]]:
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for canonical, entries in lexicon.items():
        patterns = []
        for surface, case_sensitive in entries:
            flags = 0 if case_sensitive else re.IGNORECASE
            patterns.append(re.compile(rf"(?<![A-Za-z0-9]){surface}(?![A-Za-z0-9])", flags))
        compiled[canonical] = patterns
    return compiled


_DX_PATTERNS = _compile_lexicon(DIAGNOSIS_LEXICON)
_RX_PATTERNS = _compile_lexicon(MEDICATION_LEXICON)


def _is_asserted(clause: str, start: int, end: int, *, drug: bool) -> bool:
    """Return True when the concept at ``clause[start:end]`` is an active fact."""
    before, after = clause[:start], clause[end:]
    # Only negation cues after the last contrastive conjunction apply.
    conj = list(_CONJUNCTION.finditer(before))
    scope_before = before[conj[-1].end():] if conj else before
    # NegEx-style window: a pre-trigger only reaches the six preceding tokens.
    scope_before = " ".join(scope_before.split()[-6:])
    if _PRE_NEGATION.search(scope_before):
        return False
    if _POST_NEGATION.search(after):
        return False
    if _FAMILY.search(clause):
        return False
    if drug and _ALLERGY_CONTEXT.search(clause):
        return False
    return True


def _find_concepts(note: str, patterns: dict[str, list[re.Pattern[str]]], *, drug: bool) -> list[str]:
    found: set[str] = set()
    for clause in split_clauses(note):
        for canonical, regexes in patterns.items():
            for regex in regexes:
                for match in regex.finditer(clause.text):
                    if _is_asserted(clause.text, match.start(), match.end(), drug=drug):
                        found.add(canonical)
                        break
                if canonical in found:
                    break
    return sorted(found)


def extract_diagnoses(note: str) -> list[str]:
    found = set(_find_concepts(note, _DX_PATTERNS, drug=False))
    if "type_2_diabetes" in found and _TYPE1_DIABETES.search(note) and not re.search(
        r"(?i:type (?:2|ii)|T2DM|DM2)", note
    ):
        found.discard("type_2_diabetes")
    return sorted(found)


def extract_medications(note: str) -> list[str]:
    return _find_concepts(note, _RX_PATTERNS, drug=True)


# ---------------------------------------------------------------------------
# Numeric fields
# ---------------------------------------------------------------------------
_NUM = r"(\d{1,4}(?:[.,]\d{1,3})?)"
_LINK = r"(?:\s*(?:of|was|is|at|measured|approximately|approx\.?|about|around|~|estimated(?: at)?|=|:)\s*)*\s*"


def _to_float(text: str) -> float:
    return float(text.replace(",", "."))


def _search_all(pattern: re.Pattern[str], note: str) -> list[re.Match[str]]:
    return list(pattern.finditer(note))


_HR = re.compile(
    rf"(?i:\b(?:heart rate|HR|pulse(?: rate)?|ventricular rate|herzfrequenz)\b){_LINK}(\d{{2,3}})(?!\d|[.,]\d|\s*/\s*\d)"
    r"\s*(?:bpm|/ ?min|per minute|beats)?"
)
_BP = re.compile(
    rf"(?i:\b(?:blood pressure|BP|NIBP|RR|arterial pressure)\b){_LINK}(\d{{2,3}})\s*(?:/|over)\s*(\d{{2,3}})"
)
_SBP_ONLY = re.compile(rf"(?i:\b(?:SBP|systolic (?:blood pressure|BP|pressure))\b){_LINK}(\d{{2,3}})")
_CREATININE = re.compile(
    rf"(?i:\b(?:serum |plasma |s-?)?(?:creatinine|creat|kreatinin|cr|scr)\b){_LINK}{_NUM}\s*"
    r"(?P<unit>mg/dl|mg/100 ?ml|µmol/l|μmol/l|umol/l|micromol/l|mmol/l)?",
)
_HEMOGLOBIN = re.compile(
    rf"(?i:\b(?:ha?emoglobin|hämoglobin|hgb|hb)\b){_LINK}{_NUM}\s*(?P<unit>g/dl|g/l|mmol/l|g%)?"
)
_LVEF = re.compile(
    rf"(?i:\b(?:LVEF|LV-?EF|EF|ejection fraction)\b){_LINK}(\d{{1,2}}(?:[.,]\d)?)(?:\s*(?:-|–|to)\s*(\d{{1,2}}))?\s*(?:%|per ?cent|percent)"
)


def _first_valid(values: list[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def extract_heart_rate(note: str) -> int | None:
    values = [int(m.group(1)) for m in _search_all(_HR, note)]
    return _first_valid([v if 20 <= v <= 250 else None for v in values])  # type: ignore[return-value]


def extract_systolic_bp(note: str) -> int | None:
    candidates: list[int | None] = []
    for match in _search_all(_BP, note):
        systolic, diastolic = int(match.group(1)), int(match.group(2))
        # RR also abbreviates respiratory rate in English notes; require a
        # plausible systolic/diastolic pair so ``RR 18/min`` is never read as BP.
        candidates.append(systolic if 50 <= systolic <= 280 and 20 <= diastolic < systolic else None)
    for match in _search_all(_SBP_ONLY, note):
        value = int(match.group(1))
        candidates.append(value if 50 <= value <= 280 else None)
    return _first_valid(candidates)  # type: ignore[return-value]


def _convert_creatinine(value: float, unit: str | None) -> float | None:
    unit = (unit or "").lower().replace("μ", "µ")
    if unit in {"µmol/l", "umol/l", "micromol/l"}:
        value = value / 88.4
    elif unit == "mmol/l":
        value = value * 1000.0 / 88.4
    elif not unit and value > 25:  # bare value in the µmol/L range
        value = value / 88.4
    return round(value, 2) if 0.1 <= value <= 25 else None


def _convert_hemoglobin(value: float, unit: str | None) -> float | None:
    unit = (unit or "").lower()
    if unit == "g/l":
        value = value / 10.0
    elif unit == "mmol/l":
        value = value * 1.611
    elif not unit and value > 30:  # bare value in the g/L range
        value = value / 10.0
    return round(value, 1) if 3 <= value <= 25 else None


def extract_creatinine(note: str) -> float | None:
    return _first_valid([_convert_creatinine(_to_float(m.group(1)), m.group("unit")) for m in _search_all(_CREATININE, note)])


def extract_hemoglobin(note: str) -> float | None:
    return _first_valid([_convert_hemoglobin(_to_float(m.group(1)), m.group("unit")) for m in _search_all(_HEMOGLOBIN, note)])


def extract_lvef(note: str) -> int | None:
    values: list[float | None] = []
    for match in _search_all(_LVEF, note):
        low = _to_float(match.group(1))
        value = (low + float(match.group(2))) / 2 if match.group(2) else low
        values.append(value if 5 <= value <= 90 else None)
    result = _first_valid(values)
    return int(round(result)) if result is not None else None


# ---------------------------------------------------------------------------
# Categorical fields
# ---------------------------------------------------------------------------
_SMOKING_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("never", re.compile(
        r"(?i:never[- ]smok|non-?smoker|nonsmoker|\bno tobacco\b|never used tobacco|does not smoke|doesn't smoke|"
        r"denies (?:smoking|tobacco)|no (?:history of )?smoking|lifelong non|nichtraucher|tobacco: no\b|smoking: no\b)"
    )),
    ("former", re.compile(
        r"(?i:ex-?smoker|ex smoker|former(?:ly)? (?:smoker|smoked|tobacco)|stopped smoking|quit(?:ting)? smoking|"
        r"gave up smoking|previous(?:ly)? smok|past smoker|prior smoker|quit tobacco|ex-?raucher|ehemalige[rn]? raucher|"
        r"smoked .{0,20}(?:until|ago)|abstinent)"
    )),
    ("current", re.compile(
        r"(?i:current(?:ly)? smok|actively smokes|active smoker|ongoing (?:tobacco|smoking)|smokes \d|"
        r"(?:still|daily) smok|smokes (?:cigarettes|daily|tobacco)|\braucher(?:in)?\b|tobacco: yes|smoking: yes|"
        r"\bsmoker\b)"
    )),
)

_ALLERGEN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("penicillin", re.compile(r"(?i:penicillin|amoxicillin|ampicillin|augmentin|beta-?lactam|flucloxacillin)")),
    ("nsaid", re.compile(r"(?i:nsaid|non-?steroidal|ibuprofen|diclofenac|naproxen|nsar)")),
    ("iodinated_contrast", re.compile(r"(?i:contrast|iodinated|iodine|radiocontrast|kontrastmittel)")),
)
_NO_ALLERGY = re.compile(
    r"(?i:\bnkda\b|\bnka\b|no known (?:drug )?allerg|no (?:known )?(?:medication|drug) allerg|no allergies|"
    r"allerg(?:y|ies)\s*[:=]\s*(?:none|nil|no)\b|keine (?:bekannten )?allergien|nil known)"
)


# "Smoking history not documented" / "no smoking data" mean *unknown*, not never.
_UNDOCUMENTED = re.compile(
    r"(?i:not (?:documented|recorded|available|known|assessed)|unknown|no (?:\w+ )?(?:data|information|details)|unavailable)"
)

# Generic answers inside an explicit smoking field ("Smoking: quit 2019").
_SMOKING_FALLBACK: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("former", re.compile(r"(?i:\b(?:quit|ex|former|stopped|previous|past)\b)")),
    ("current", re.compile(r"(?i:\b(?:current|yes|active|daily|pack)\b)")),
    ("never", re.compile(r"(?i:\b(?:never|none|no|nil|denies)\b)")),
)


def extract_smoking_status(note: str) -> str | None:
    clauses = [c.text for c in split_clauses(note) if re.search(r"(?i:smok|tobacco|cigar|raucher|nicotin)", c.text)]
    clauses = [c for c in clauses if not _FAMILY.search(c) and not _UNDOCUMENTED.search(c)]
    for rules in (_SMOKING_RULES, _SMOKING_FALLBACK):
        for clause in clauses:
            for status, pattern in rules:
                if pattern.search(clause):
                    return status
    return None


def extract_allergy(note: str) -> str | None:
    clauses = [c.text for c in split_clauses(note) if _ALLERGY_CONTEXT.search(c.text)]
    for clause in clauses:
        for allergen, pattern in _ALLERGEN_RULES:
            if pattern.search(clause) and not _NO_ALLERGY.search(clause):
                return allergen
    for clause in clauses:
        if _NO_ALLERGY.search(clause):
            return "none"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
FIELD_EXTRACTORS: dict[str, Callable[[str], Any]] = {
    "diagnoses": extract_diagnoses,
    "medications": extract_medications,
    "heart_rate_bpm": extract_heart_rate,
    "systolic_bp_mmhg": extract_systolic_bp,
    "creatinine_mg_dl": extract_creatinine,
    "hemoglobin_g_dl": extract_hemoglobin,
    "lvef_percent": extract_lvef,
    "smoking_status": extract_smoking_status,
    "allergy": extract_allergy,
}


def extract_clinical_data(note: str) -> dict[str, Any]:
    """Extract all canonical fields; every field is always present."""
    result: dict[str, Any] = {}
    for field, extractor in FIELD_EXTRACTORS.items():
        try:
            result[field] = extractor(note)
        except Exception:  # pragma: no cover - defensive: never fail a whole case
            result[field] = [] if field in ("diagnoses", "medications") else None
    return result
