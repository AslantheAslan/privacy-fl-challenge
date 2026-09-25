"""Context-aware, rule-based PII detector for synthetic clinical notes.

Design
------
Detection runs in independent passes, each producing ``(priority, Span)``
candidates. A final greedy resolver keeps the highest-priority, longest
non-overlapping spans. Passes, in priority order:

1. ``EMAIL``         – RFC-like address pattern.
2. ``PHONE_NUMBER``  – international (``+CC ...``) numbers, or national numbers
                        preceded by a phone cue.
3. ``PATIENT_ID``    – known site formats, generic prefixed identifiers and
                        identifiers following an ID cue (``MRN``, ``UHID``...).
4. dates             – many surface formats; labelled ``DATE_OF_BIRTH`` or
                        ``ENCOUNTER_DATE`` from the nearest preceding cue, with
                        an age/year consistency fallback for cue-less dates.
5. ``ADDRESS``       – anchored on a postal code + city (German 5-digit PLZ or
                        Indian 6-digit PIN), expanded left to the field boundary;
                        plus a cue-only fallback (``Address:``, ``Residence``).
6. names             – person names after patient/clinician cues or titles, a
                        structural "name, born ..." pattern, names recovered
                        from clinician e-mail local parts, and finally
                        propagation of every confirmed name to its repeated
                        occurrences (the dictionary labels each repetition).

Rules are deliberately explicit so that every redaction can be traced back to
the rule that fired (``Span.source``), which matters for auditability in a
clinical setting. Over-redaction is controlled by requiring either a cue or a
highly specific shape for every label; clinical eponyms (``Parkinson disease``)
and brands (``Eliquis``) are never matched because names need two Title-case
tokens *and* a person cue.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from .spans import Span, resolve_overlaps

# ---------------------------------------------------------------------------
# Character classes and reusable fragments
# ---------------------------------------------------------------------------
_UP = "A-ZÄÖÜÀ-ÖØ-Ý"
_LO = "a-zäöüßà-öø-ÿ"
_NAME_TOKEN = rf"(?:[{_UP}]['’][{_UP}][{_LO}]+|[{_UP}][{_LO}]+(?:[-'’][{_UP}]?[{_LO}]+)*)"
_PARTICLE = r"(?:van|von|der|den|de|da|del|di|du|le|la|bin|al|el|ter|zu)"
_NAME = rf"{_NAME_TOKEN}(?: (?:{_PARTICLE} )?{_NAME_TOKEN}){{1,3}}"
_TITLE = r"(?:Dr|Prof|Mr|Mrs|Ms|Miss|Mx|Frau|Herr|Smt)\.?"
_TITLES = rf"(?:(?:{_TITLE}|med\.|rer\. nat\.)[ \t]+){{0,3}}"
_CAPS_TOKEN = rf"[{_UP}]{{2,}}(?:-[{_UP}]{{2,}})?"
# Person names after an explicit cue may also be inverted ("Neumann, Anton") or
# carry an upper-case surname ("NEUMANN Anton" / "Anton NEUMANN").
_PERSON = (
    rf"(?:{_NAME}|{_CAPS_TOKEN},? {_NAME_TOKEN}(?: {_NAME_TOKEN})?|{_NAME_TOKEN} {_CAPS_TOKEN}"
    rf"|{_NAME_TOKEN}, {_NAME_TOKEN}(?: {_NAME_TOKEN})?)"
)

# Capitalised words that may follow a name on the same line and must never be
# absorbed into it, plus structural words that can never be person names.
_NAME_STOPWORDS = {
    "patient", "name", "dob", "born", "case", "id", "address", "telephone", "phone", "mobile",
    "encounter", "admission", "diagnoses", "diagnosis", "known", "medication", "medications",
    "medicines", "allergies", "allergy", "smoking", "signed", "reviewed", "consultant", "home",
    "residence", "visit", "contact", "treating", "laboratory", "echocardiography", "on", "the",
    "no", "family", "tab", "rx", "dx", "obs", "problem", "author", "electronically", "for",
    "current", "active", "vitals", "measurements", "clinical", "examination", "age", "sex",
    "mrn", "uhid", "email", "mail", "doctor", "physician", "clinician", "attending", "dr",
    "prof", "summary", "note", "record", "synthetic", "discharge", "berlin", "chennai",
    "hyderabad", "flat", "street", "road", "lane", "avenue", "colony", "hospital", "clinic",
    "department", "ward", "unit", "date", "seen", "responsible", "and", "with", "of", "in",
    "pt", "admitted",
}
_TITLE_WORDS = {"dr", "prof", "mr", "mrs", "ms", "miss", "mx", "frau", "herr", "smt"}

_MONTHS = (
    r"(?i:jan(?:uary|uar)?|feb(?:ruary|ruar)?|m(?:ar(?:ch)?|ärz|aerz)|apr(?:il)?|ma[yi]|jun[ei]?|"
    r"jul[yi]?|aug(?:ust)?|sep(?:t(?:ember)?)?|o[ck]t(?:ober)?|nov(?:ember)?|de[cz](?:ember)?)"
)
_YEAR4 = r"(?:19|20)\d{2}"

_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"(?<![\d/.-])\d{{1,2}}[./-]\d{{1,2}}[./-]{_YEAR4}(?![\d/-])"),
    re.compile(rf"(?<![\d/.-]){_YEAR4}[./-]\d{{1,2}}[./-]\d{{1,2}}(?![\d/-])"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\.?[ -]?{_MONTHS}\.?[ ,-]+{_YEAR4}\b"),
    re.compile(rf"\b{_MONTHS}\.? \d{{1,2}}(?:st|nd|rd|th)?,? {_YEAR4}\b"),
)
# Two-digit-year dates are ambiguous with ratios, so they need an explicit cue.
_SHORT_DATE = re.compile(r"(?<![\d/.-])\d{1,2}[./]\d{1,2}[./]\d{2}(?![\d/-]|\.\d)")
_STRICT_SHORT_DATE = re.compile(r"\d{2}([./])\d{2}\1\d{2}")

_DOB_CUE = re.compile(
    r"(?i:\b(?:dob|d\.o\.b\.?|date of birth|birth ?date|born|geb(?:oren|\.)?|birthday)(?![a-z]))"
)
_ENC_CUE = re.compile(
    r"(?i:\b(?:encounter|admission|admitted|adm|doa|visit|seen|date of (?:visit|admission|service|encounter)|"
    r"discharged?|dod|consultation|presented|attendance|service date|aufnahme|date|on)(?![a-z]))"
)

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")

_PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ .\-/]?\(?\d{1,5}\)?)(?:[ .\-/]?\d{2,6}){1,5}(?![\d])")
_PHONE_CUE = r"(?i:tel(?:ephone|\.)?|phone|ph|mobile|mob\.?|cell|contact|fon|handy)"
_PHONE_NATIONAL = re.compile(
    rf"{_PHONE_CUE}\s*(?:(?i:no\.?|number|nr\.?|#))?\s*[:.]?\s*"
    r"(?P<num>\(?\d{2,5}\)?(?:[ .\-/]?\d{2,8}){1,4})(?!\d)"
)

_SITE_ID = re.compile(
    r"\b(?:B-\d{6}|CHN-\d{7}|HYD\d{6}|BER/\d{4}/\d{2}|UHID/\d{5}/\d{2}|MRN-\d{2}-\d{5})\b"
)
_GENERIC_ID = re.compile(r"(?<![\w/+-])[A-Z]{1,6}(?:[-/.]?\d+){1,4}(?![\w/-])")
_ID_CUE = re.compile(
    r"(?i:\b(?:mrn|uhid|case id|case no\.?|patient id|pat\.? ?id|hospital (?:id|no\.?)|record (?:id|no\.?|number)|"
    r"reg(?:istration)? no\.?|ip no\.?|fall-?nr\.?|id)\b)\s*[:#.]?\s*(?P<id>[A-Z0-9][A-Za-z0-9/-]{3,}[0-9])"
)

_DE_POSTAL = re.compile(rf"(?:D-)?\d{{5}} [{_UP}][{_LO}]+(?:[ -][{_UP}][{_LO}]+)?")
_IN_POSTAL = re.compile(rf"[{_UP}][{_LO}]+(?: [{_UP}][{_LO}]+)?(?: ?[-–] ?| )\d{{3}} ?\d{{3}}(?!\d)")
_ADDRESS_CUE = re.compile(
    r"(?i:\b(?:address|residence|residential address|home address|home|resides at|lives at|living at|"
    r"anschrift|wohnort|wohnhaft in|from)\b)\s*[:=-]?\s*"
)
_ABBREVIATIONS_WITH_DOT = {"no", "nr", "str", "st", "rd", "ave", "apt", "bldg", "opp", "nagar"}

_PATIENT_CUE = re.compile(
    rf"(?i:\b(?:patient(?:'s)? name|patientin|patient|full name|name|pt)\b\.?)[ \t]*[:=\-–]?[ \t]*{_TITLES}(?P<name>{_PERSON})"
)
_CLINICIAN_CUE = re.compile(
    r"(?i:\b(?:attending physician|attending|treating clinician|treating physician|treating doctor|clinician|"
    r"physician|consultant|responsible (?:doctor|physician|clinician)|doctor|electronically signed by|signed by|"
    r"signed|reviewed by|verified by|approved by|author|seen by|examined by|treated by|discharged by|"
    r"dictated by|referring (?:doctor|physician)|referred by|resident|cc)\b)"
    rf"[ \t]*[:=\-–]?[ \t]*{_TITLES}(?P<name>{_PERSON})"
)
_CLINICIAN_TITLE = re.compile(rf"\b(?:Dr|Prof)\.?[ \t]+(?:(?:med|Dr|rer\. nat)\.?[ \t]+)*(?P<name>{_PERSON})")
_STRUCTURAL_PATIENT = re.compile(
    rf"(?<![\w'’.-])(?P<name>{_PERSON})"
    rf"(?=\s*(?:,\s*(?i:born|dob|d\.o\.b|date of birth|geb)|\(\s*(?i:dob|d\.o\.b)|,\s*\d{{1,3}}\s*y|\s*/\s*[A-Z]{{1,6}}[-/]?\d))"
)


def _fold(text: str) -> str:
    """Case-fold and transliterate umlauts the way e-mail local parts do."""
    text = text.lower()
    for src, dst in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _trim_name(note: str, start: int, end: int) -> tuple[int, int] | None:
    """Strip leading cue/title tokens and anything from the first stop-word on.

    Requires at least two capitalised tokens to remain.
    """
    tokens = list(re.finditer(r"\S+", note[start:end]))
    while tokens and tokens[0].group(0).lower().strip(".,;:") in _NAME_STOPWORDS | _TITLE_WORDS:
        tokens.pop(0)
    kept: list[re.Match[str]] = []
    for token in tokens:
        if token.group(0).lower().strip(".,;:") in _NAME_STOPWORDS:
            break
        kept.append(token)
    capitalised = [t for t in kept if t.group(0)[0].isupper()]
    if len(capitalised) < 2:
        return None
    return start + kept[0].start(), start + kept[-1].end()


def _valid_date(text: str) -> bool:
    numbers = [int(part) for part in re.findall(r"\d+", text)]
    if len(numbers) == 3:
        if numbers[0] > 1900:  # ISO yyyy-mm-dd
            year, month, day = numbers
        else:
            day, month, year = numbers
        return 1 <= month <= 12 and 1 <= day <= 31
    return True  # month-name formats are validated by the regex itself


@dataclass
class _DateCandidate:
    start: int
    end: int
    year: int
    label: str | None


class PiiDetector:
    """Detect PII spans in a clinical note.

    ``age_years`` (from the structured features) is optional and only used to
    disambiguate cue-less dates, e.g. a bare date in a document header.
    """

    def detect(self, note: str, age_years: int | None = None) -> list[Span]:
        candidates: list[tuple[int, Span]] = []
        candidates += [(0, s) for s in self._emails(note)]
        candidates += [(1, s) for s in self._phones(note)]
        candidates += [(2, s) for s in self._ids(note)]
        candidates += [(3, s) for s in self._dates(note, age_years)]
        candidates += [(4, s) for s in self._addresses(note)]
        structural = resolve_overlaps(candidates)
        names = self._names(note, structural)
        return resolve_overlaps([(0, s) for s in structural] + [(5, s) for s in names])

    # -- contact details -----------------------------------------------------
    @staticmethod
    def _emails(note: str) -> Iterable[Span]:
        for match in _EMAIL.finditer(note):
            yield Span(match.start(), match.end(), "EMAIL", "email")

    @staticmethod
    def _phones(note: str) -> Iterable[Span]:
        for match in _PHONE_INTL.finditer(note):
            if 8 <= sum(ch.isdigit() for ch in match.group(0)) <= 15:
                yield Span(match.start(), match.end(), "PHONE_NUMBER", "phone_intl")
        for match in _PHONE_NATIONAL.finditer(note):
            number = match.group("num")
            if 7 <= sum(ch.isdigit() for ch in number) <= 15 and not _valid_date_like(number):
                yield Span(match.start("num"), match.end("num"), "PHONE_NUMBER", "phone_cue")

    # -- identifiers -----------------------------------------------------------
    @staticmethod
    def _ids(note: str) -> Iterable[Span]:
        for match in _SITE_ID.finditer(note):
            yield Span(match.start(), match.end(), "PATIENT_ID", "id_site_format")
        for match in _GENERIC_ID.finditer(note):
            if sum(ch.isdigit() for ch in match.group(0)) >= 5:
                yield Span(match.start(), match.end(), "PATIENT_ID", "id_generic")
        for match in _ID_CUE.finditer(note):
            value = match.group("id")
            if sum(ch.isdigit() for ch in value) >= 4 and not any(p.fullmatch(value) for p in _DATE_PATTERNS):
                yield Span(match.start("id"), match.end("id"), "PATIENT_ID", "id_cue")

    # -- dates -----------------------------------------------------------------
    def _dates(self, note: str, age_years: int | None) -> list[Span]:
        found: list[_DateCandidate] = []
        occupied: list[tuple[int, int]] = []
        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(note):
                if any(match.start() < e and s < match.end() for s, e in occupied):
                    continue
                if not _valid_date(match.group(0)):
                    continue
                year = int(re.findall(r"(?:19|20)\d{2}", match.group(0))[-1])
                occupied.append((match.start(), match.end()))
                found.append(_DateCandidate(match.start(), match.end(), year, None))
        for match in _SHORT_DATE.finditer(note):
            if any(match.start() < e and s < match.end() for s, e in occupied):
                continue
            left = note[max(0, match.start() - 25) : match.start()]
            has_cue = bool(_DOB_CUE.search(left) or _ENC_CUE.search(left))
            if (has_cue or _STRICT_SHORT_DATE.fullmatch(match.group(0))) and _valid_date(match.group(0)):
                yy = int(match.group(0)[-2:])
                found.append(_DateCandidate(match.start(), match.end(), 2000 + yy if yy < 40 else 1900 + yy, None))
        found.sort(key=lambda d: d.start)

        previous_end = 0
        for date in found:
            window_start = max(previous_end, date.start - 40)
            window = note[window_start : date.start]
            last_dob = max((m.end() for m in _DOB_CUE.finditer(window)), default=-1)
            last_enc = max((m.end() for m in _ENC_CUE.finditer(window)), default=-1)
            if last_dob >= 0 and last_dob >= last_enc:
                date.label = "DATE_OF_BIRTH"
            elif last_enc > last_dob:
                date.label = "ENCOUNTER_DATE"
            previous_end = date.end

        self._resolve_cueless_dates(found, age_years)
        return [Span(d.start, d.end, d.label or "ENCOUNTER_DATE", "date") for d in found]

    @staticmethod
    def _resolve_cueless_dates(dates: list[_DateCandidate], age_years: int | None) -> None:
        unresolved = [d for d in dates if d.label is None]
        if not unresolved:
            return
        has_dob = any(d.label == "DATE_OF_BIRTH" for d in dates)
        reference_year = max(d.year for d in dates)
        for date in unresolved:
            if has_dob:
                date.label = "ENCOUNTER_DATE"
                continue
            implied_age = reference_year - date.year
            if age_years is not None and abs(implied_age - age_years) <= 1 and implied_age > 0:
                date.label = "DATE_OF_BIRTH"
            elif age_years is None and implied_age >= 16:
                date.label = "DATE_OF_BIRTH"
            else:
                date.label = "ENCOUNTER_DATE"

    # -- addresses ---------------------------------------------------------------
    def _addresses(self, note: str) -> list[Span]:
        spans: list[Span] = []
        anchors = [(m.start(), m.end()) for m in _DE_POSTAL.finditer(note)]
        anchors += [(m.start(), m.end()) for m in _IN_POSTAL.finditer(note)]
        for anchor_start, anchor_end in sorted(anchors):
            start = self._address_left_boundary(note, anchor_start)
            if start is None or anchor_end - start > 120:
                continue
            spans.append(Span(start, anchor_end, "ADDRESS", "address_postal"))
        if not spans:
            for match in _ADDRESS_CUE.finditer(note):
                start = match.end()
                stop = re.search(r"\s*(?:\||;|\n|\.\s+[A-Z]|$)", note[start:])
                end = start + (stop.start() if stop else 0)
                segment = note[start:end]
                if 8 <= len(segment) <= 120 and re.search(r"\d", segment) and "," in segment:
                    spans.append(Span(start, end, "ADDRESS", "address_cue"))
        return spans

    @staticmethod
    def _address_left_boundary(note: str, anchor_start: int) -> int | None:
        """Walk left from the postal anchor to the start of the address field."""
        position = anchor_start
        while position > 0 and anchor_start - position < 110:
            ch = note[position - 1]
            if ch in "\n|;:()[]<>=":
                break
            if ch == "." and position < len(note) and note[position] == " ":
                word = re.search(r"(\w+)\.$", note[:position])
                if not (word and word.group(1).lower() in _ABBREVIATIONS_WITH_DOT):
                    break
            position -= 1
        segment = note[position:anchor_start]
        cue = _ADDRESS_CUE.match(segment.lstrip())
        offset = len(segment) - len(segment.lstrip())
        if cue:
            offset += cue.end()
        start = position + offset
        # A street address always contains a house/flat number before the anchor.
        if not re.search(r"\d", note[start:anchor_start]):
            return None
        return start

    # -- names -------------------------------------------------------------------
    def _names(self, note: str, taken: list[Span]) -> list[Span]:
        def free(start: int, end: int) -> bool:
            return not any(start < s.end and s.start < end for s in taken)

        found: dict[tuple[int, int], tuple[str, str]] = {}

        def add(start: int, end: int, label: str, source: str, override: bool = False) -> None:
            trimmed = _trim_name(note, start, end)
            if trimmed is None or not free(*trimmed):
                return
            if trimmed in found and not override:
                return
            found[trimmed] = (label, source)

        for match in _PATIENT_CUE.finditer(note):
            add(match.start("name"), match.end("name"), "PATIENT_NAME", "patient_cue")
        for match in _STRUCTURAL_PATIENT.finditer(note):
            add(match.start("name"), match.end("name"), "PATIENT_NAME", "patient_structural")
        # Clinician evidence overrides patient evidence for the same span.
        for match in _CLINICIAN_CUE.finditer(note):
            add(match.start("name"), match.end("name"), "CLINICIAN_NAME", "clinician_cue", override=True)
        for match in _CLINICIAN_TITLE.finditer(note):
            add(match.start("name"), match.end("name"), "CLINICIAN_NAME", "clinician_title", override=True)
        for start, end in self._names_from_emails(note):
            add(start, end, "CLINICIAN_NAME", "clinician_email", override=True)

        # Propagate every confirmed name string to its other occurrences.
        confirmed = {note[s:e]: label for (s, e), (label, _) in found.items()}
        for text, label in confirmed.items():
            for match in re.finditer(rf"(?<![\w]){re.escape(text)}(?![\w])", note):
                if (match.start(), match.end()) not in found:
                    add(match.start(), match.end(), label, "propagated")

        spans = [Span(s, e, label, source) for (s, e), (label, source) in found.items()]
        return resolve_overlaps((0, span) for span in spans)

    @staticmethod
    def _names_from_emails(note: str) -> Iterable[tuple[int, int]]:
        """Recover ``Laura König`` from ``laura.koenig@...`` anywhere in the note."""
        for match in _EMAIL.finditer(note):
            local = match.group(0).split("@", 1)[0]
            parts = [p for p in re.split(r"[._-]+", local) if len(p) >= 2 and not p.isdigit()]
            if len(parts) < 2:
                continue
            target = [_fold(p) for p in parts]
            for candidate in re.finditer(_NAME, note):
                tokens = list(re.finditer(r"\S+", candidate.group(0)))
                for i in range(len(tokens) - len(target) + 1):
                    window = tokens[i : i + len(target)]
                    if [_fold(t.group(0)) for t in window] == target:
                        yield candidate.start() + window[0].start(), candidate.start() + window[-1].end()


def _valid_date_like(text: str) -> bool:
    return any(pattern.fullmatch(text) for pattern in _DATE_PATTERNS)


_DEFAULT_DETECTOR = PiiDetector()


def detect_pii(note: str, age_years: int | None = None) -> list[Span]:
    """Module-level convenience wrapper around :class:`PiiDetector`."""
    return _DEFAULT_DETECTOR.detect(note, age_years=age_years)
