"""Advisor rules: risk level → advice, and when to send an SMS alert.

IMPORTANT: the advice text is a draft. It must be checked by the Municipal
Agriculture Office (correct advice, correct varieties) and a native speaker for
each language before any farmer receives it. VALIDATED stays False until then.
"""
from __future__ import annotations

from datetime import date

VALIDATED = False
LANGS = ("tl", "ceb", "en")  # Tagalog, Bisaya (Cebuano), English
SMS_LIMIT = 160              # one SMS segment; longer messages cost more and may split

# One short alert per level. {area} = barangay or municipality.
ALERTS = {
    "High": {
        "tl": "BABALA EL NIÑO: Mataas ang panganib ng tagtuyot sa {area}. Ipagpaliban muna ang pagtanim. Reply TIPS para sa payo, STOP para huminto.",
        "ceb": "PASIDAAN EL NIÑO: Taas ang peligro sa hulaw sa {area}. Ayaw sa una pagtanom. Reply TIPS para tambag, STOP aron mohunong.",
        "en": "EL NIÑO ALERT: High drought risk in {area}. Delay planting for now. Reply TIPS for advice, STOP to unsubscribe.",
    },
    "Moderate": {
        "tl": "PAALALA EL NIÑO: Katamtaman ang panganib ng tagtuyot sa {area}. Mag-ipon ng tubig. Reply TIPS para sa payo, STOP para huminto.",
        "ceb": "PAHINUMDOM EL NIÑO: Kasarangan ang peligro sa hulaw sa {area}. Pagtigom og tubig. Reply TIPS para tambag, STOP aron mohunong.",
        "en": "EL NIÑO NOTICE: Moderate drought risk in {area}. Start storing water. Reply TIPS for advice, STOP to unsubscribe.",
    },
    "Low": {
        "tl": "EL NIÑO UPDATE: Mababa na ang panganib ng tagtuyot sa {area}. Maaari nang magtanim ayon sa karaniwang iskedyul.",
        "ceb": "EL NIÑO UPDATE: Ubos na ang peligro sa hulaw sa {area}. Pwede na magtanom sumala sa naandang iskedyul.",
        "en": "EL NIÑO UPDATE: Drought risk in {area} is now low. You can plant on your usual schedule.",
    },
}

# Replies to the TIPS keyword. [VARIETY] and [TECHNICIAN] are filled in per municipality.
TIPS = {
    "High": {
        "tl": "PAYO: 1) Ipagpaliban ang pagtanim. 2) Gumamit ng binhing matibay sa tagtuyot: [VARIETY]. 3) Kumonsulta kay [TECHNICIAN].",
        "ceb": "TAMBAG: 1) Ayaw sa una pagtanom. 2) Gamita ang binhi nga lig-on sa hulaw: [VARIETY]. 3) Pakigsulti kang [TECHNICIAN].",
        "en": "TIPS: 1) Delay planting. 2) Use a drought-tolerant variety: [VARIETY]. 3) Talk to [TECHNICIAN].",
    },
    "Moderate": {
        "tl": "PAYO: 1) Mag-ipon ng tubig. 2) Ihanda ang binhing matibay sa tagtuyot: [VARIETY]. 3) Bantayan ang susunod na abiso.",
        "ceb": "TAMBAG: 1) Pagtigom og tubig. 2) Andama ang binhi nga lig-on sa hulaw: [VARIETY]. 3) Hulata ang sunod nga pahibalo.",
        "en": "TIPS: 1) Store water. 2) Get a drought-tolerant variety ready: [VARIETY]. 3) Watch for the next update.",
    },
    "Low": {
        "tl": "PAYO: Walang espesyal na hakbang ngayon. Sundin ang karaniwang iskedyul ng pagtanim.",
        "ceb": "TAMBAG: Walay espesyal nga lakang karon. Sunda ang naandang iskedyul sa pagtanom.",
        "en": "TIPS: No special steps needed now. Follow your usual planting schedule.",
    },
}

ORDER = {"Low": 0, "Moderate": 1, "High": 2}
REMIND_HIGH_EVERY_DAYS = 30  # a farmer still at High gets a reminder at most this often


def should_alert(previous: str | None, current: str, last_sent: date | None, today: date) -> bool:
    """Send when the risk goes UP, an all-clear when it drops to Low after a warning,
    and a reminder while it stays High — never daily repeats of the same level."""
    if previous is None:
        return current != "Low"            # first reading: only warn if there is something to warn about
    if ORDER[current] > ORDER[previous]:
        return True                         # risk went up
    if current == "Low" and previous != "Low":
        return True                         # all clear
    if current == "High" and last_sent is not None:
        return (today - last_sent).days >= REMIND_HIGH_EVERY_DAYS
    return False


def alert_text(level: str, lang: str, area: str) -> str:
    return ALERTS[level][lang].format(area=area)


def tips_text(level: str, lang: str) -> str:
    return TIPS[level][lang]
