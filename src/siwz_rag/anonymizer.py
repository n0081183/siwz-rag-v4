"""Anonimizacja nazw producentów i produktów SecOps w odpowiedziach.

Używane gdy generujemy wymagania SIWZ (mają być vendor-neutral) lub jako opcja
przy verify żeby raport mógł być przekazany dalej bez bias.

Lista wzorców jest jawnie uporządkowana: dłuższe/specyficzne nazwy MUSZĄ być przed
krótszymi, żeby "Cortex XDR" nie został zamieniony na "platforma SecOps XDR".
"""

from __future__ import annotations

import re

# Mapa: regex pattern → neutralny opis. KOLEJNOŚĆ MA ZNACZENIE.
BRAND_MAP: list[tuple[str, str]] = [
    (r"Palo Alto Networks", "dostawca rozwiązania"),
    (r"Cortex XDR", "platforma XDR"),
    (r"Cortex XSIAM", "platforma SIEM/SOAR nowej generacji"),
    (r"Cortex XSOAR", "platforma SOAR"),
    (r"Cortex Xpanse", "platforma ASM"),
    (r"Cortex Data Lake", "chmurowe jezioro danych bezpieczeństwa"),
    (r"Cortex", "platforma SecOps"),
    (r"XSIAM", "platforma SIEM/SOAR nowej generacji"),
    (r"XSOAR", "platforma SOAR"),
    (r"XPANSE", "platforma ASM"),
    (r"XDR", "platforma XDR"),
    (r"WildFire", "silnik analizy złośliwego oprogramowania (sandbox)"),
    (r"Strata", "platforma zarządzania politykami bezpieczeństwa"),
    (r"PAN-OS", "dedykowany system operacyjny urządzenia sieciowego"),
    (r"AutoFocus", "usługa threat intelligence"),
    (r"Unit 42", "centrum badań zagrożeń"),
    (r"Prisma Cloud", "platforma bezpieczeństwa chmury"),
    (r"Prisma Access", "usługa SASE/SSE"),
    (r"Panorama", "centralny system zarządzania politykami"),
    (r"NGFW", "zapora nowej generacji (NGFW)"),
    (r"Traps", "agent ochrony punktów końcowych"),
]

_COMPILED = [(re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in BRAND_MAP]


def anonymize(text: str) -> str:
    """Zamień nazwy producentów/produktów na neutralne opisy.

    Idempotentne — wielokrotne wywołanie zwraca ten sam wynik.
    """
    if not text:
        return text
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text
