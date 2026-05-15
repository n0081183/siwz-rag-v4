"""Wiedza domenowa — terminologia SIWZ ↔ Cortex.

v4 vs v3:
  v3 zawierał ogromną statyczną macierz kompatybilności OS w `ARCHITECTURE_FACTS["Compatibility"]`,
  bo ingest z PDF rozsypywał tabele kompatybilności i nie sposób było polegać na retrievalu.

  v4 ma HTML-aware chunker z ATOMOWYMI tabelami → tabela "Windows Desktop" trafia do
  Qdrant jako jeden chunk z pełnymi nagłówkami kolumn. Reranker bge-v2-m3 zapewnia,
  że trafia do top-K. Dlatego statyczne dane kompatybilności w product_knowledge stały się
  redundantne i potencjalnie szkodliwe (gdy dokumentacja online się zmieni, statyczny PK
  staje się stary i zacznie konfliktować z retrievalem).

  Co zostało:
    1. ARCHITECTURE_CONCEPTS — krótkie opisy "co to za produkt i z czego się składa".
       Nie zmienia się prawie wcale (Cortex XSIAM zawsze będzie SIEM/SOAR SaaS).
       Wstrzykiwane w system prompt — daje modelowi "kotwicę" co to za produkt.
    2. TERM_MAPPING — mapowanie polskich/angielskich terminów SIWZ na Cortex
       (broker VM = węzeł pośredniczący, WildFire = sandbox itp.).
       Używane do query expansion i jako "podpowiedź" dla LLM w prompt'cie.
    3. SYNONYMS — synonimy domenowe do query expansion przed retrievalem.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# WARSTWA 1: Architektura — STABILNE, koncepcyjne opisy produktów
# Brak danych "ile wersji wspiera co". To leci z retrievalu.
# ═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE_CONCEPTS: Dict[str, Dict[str, str]] = {
    "XSIAM": {
        "pl": (
            "Cortex XSIAM to platforma SIEM/SOAR/XDR nowej generacji działająca w modelu SaaS (chmura). "
            "Łączy SIEM, SOAR, XDR, ASM i TIP w jednym produkcie. "
            "Architektura: tenant w chmurze + opcjonalny Broker VM (węzeł pośredniczący) on-prem. "
            "Broker VM pośredniczy między agentami/źródłami danych a chmurą XSIAM (syslog, buforowanie offline). "
            "XSIAM korzysta z Cortex Data Lake jako wspólnego jeziora danych. "
            "Agenty: Cortex XDR Agent (ten sam co w XDR standalone). "
            "WildFire (sandbox SaaS) jest zintegrowany. "
            "Wbudowane: Copilot (chatbot AI), Machine Learning analytics. "
            "REST API do pełnej automatyzacji. Playbooki SOAR wbudowane."
        ),
        "en": (
            "Cortex XSIAM is a next-gen SIEM/SOAR/XDR platform running as SaaS (cloud). "
            "Combines SIEM, SOAR, XDR, ASM, and TIP in one product. "
            "Architecture: cloud tenant + optional on-prem Broker VM (intermediary node). "
            "Broker VM mediates between agents/data sources and the XSIAM cloud (syslog, offline buffering). "
            "Uses Cortex Data Lake as unified data lake. "
            "Agents: Cortex XDR Agent (same as XDR standalone). "
            "WildFire (SaaS sandbox) is integrated. "
            "Built-in: Copilot (AI chatbot), ML analytics. "
            "REST API for full automation. SOAR playbooks built-in."
        ),
    },
    "XDR": {
        "pl": (
            "Cortex XDR to platforma Extended Detection and Response (EDR/XDR). "
            "Działa jako SaaS (cloud) lub on-prem (Cortex XDR on-prem). "
            "Cortex XDR Agent instaluje się na endpointach (Windows, macOS, Linux, Android, iOS). "
            "Agent zapewnia: EDR/EPP, telemetrię, response actions, kontrolę USB, ochronę przed exploitami, "
            "ransomware'em i malware'em, host firewall. "
            "Broker VM pośredniczy w komunikacji agent↔chmura. "
            "WildFire wykonuje analizę dynamiczną plików (sandbox). "
            "Konsola zarządzania przez przeglądarkę (HTML5, bez pluginów). "
            "Integracja z NGFW Palo Alto, Panorama, Prisma."
        ),
        "en": (
            "Cortex XDR is an Extended Detection and Response (EDR/XDR) platform. "
            "Runs as SaaS (cloud) or on-prem (Cortex XDR on-prem). "
            "Cortex XDR Agent installed on endpoints (Windows, macOS, Linux, Android, iOS). "
            "Agent provides: EDR/EPP, telemetry, response actions, USB device control, exploit/ransomware/"
            "malware protection, host firewall. "
            "Broker VM mediates agent↔cloud traffic. "
            "WildFire performs dynamic file analysis (sandbox). "
            "Browser-based console (HTML5, no plugins). "
            "Integration with Palo Alto NGFW, Panorama, Prisma."
        ),
    },
    "XSOAR": {
        "pl": (
            "Cortex XSOAR to platforma SOAR (Security Orchestration, Automation and Response). "
            "Wersja XSOAR 6 jest on-prem (instalacja u klienta), wersja XSOAR 8 jest SaaS. "
            "Funkcje: playbooki, automatyzacja incydentów, case management, War Room (współpraca analityków). "
            "Integracje: kilkaset paczek (content packs) z gotowymi integracjami z produktami SOC. "
            "W Cortex XSIAM funkcjonalność SOAR jest wbudowana — XSOAR pozostaje osobnym produktem dla klientów "
            "którzy chcą stand-alone."
        ),
        "en": (
            "Cortex XSOAR is a SOAR (Security Orchestration, Automation and Response) platform. "
            "XSOAR 6 is on-prem (installed at customer site), XSOAR 8 is SaaS. "
            "Features: playbooks, incident automation, case management, War Room (analyst collaboration). "
            "Integrations: hundreds of content packs with ready-made integrations to SOC products. "
            "In Cortex XSIAM SOAR is built-in — XSOAR remains a separate product for customers who want "
            "stand-alone SOAR."
        ),
    },
    "XPANSE": {
        "pl": (
            "Cortex Xpanse to platforma Attack Surface Management (ASM). "
            "Działa jako SaaS. Skanuje publiczną przestrzeń adresową IPv4/IPv6 organizacji, "
            "wykrywa zasoby (domeny, IP, certyfikaty, usługi), rozpoznaje technologie, frameworki, CMS, wtyczki. "
            "Testuje pod kątem podatności i błędów konfiguracyjnych. "
            "Funkcjonalność Xpanse jest wbudowana w Cortex XSIAM."
        ),
        "en": (
            "Cortex Xpanse is an Attack Surface Management (ASM) platform. "
            "Runs as SaaS. Scans the organization's public IPv4/IPv6 address space, "
            "discovers assets (domains, IPs, certificates, services), identifies technologies, frameworks, "
            "CMS, plugins. Tests for vulnerabilities and misconfigurations. "
            "Xpanse functionality is built into Cortex XSIAM."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# WARSTWA 2: Term mapping SIWZ ↔ Cortex
# Klucz = termin z SIWZ (PL lub EN), wartość = co to jest w Cortex.
# ═══════════════════════════════════════════════════════════════════════════════

TERM_MAPPING: Dict[str, str] = {
    # Agenty i endpointy
    "agent EDR": "Cortex XDR Agent (EDR/EPP)",
    "agent XDR": "Cortex XDR Agent",
    "agent antywirusowy": "Cortex XDR Agent (malware protection)",
    "ochrona endpointów": "Cortex XDR Agent (Endpoint Protection)",
    "ochrona przed exploitami": "Cortex XDR Exploit Protection",
    "ochrona przed ransomware": "Cortex XDR Behavioral Threat Protection",

    # Architektura
    "węzeł pośredniczący": "Broker VM",
    "broker": "Broker VM",
    "jezioro danych": "Cortex Data Lake",
    "wspólny zbiór danych": "Cortex Data Lake",
    "sandbox": "WildFire",
    "system typu sandbox": "WildFire",
    "architektura chmurowa": "SaaS / Cloud-native",
    "model chmurowy": "SaaS deployment",
    "wersja chmurowa": "Cortex XDR / XSIAM SaaS",
    "wersja on-prem": "Cortex XDR on-prem / XSOAR 6 on-prem",

    # SIEM/SOAR/XDR
    "SIEM": "Cortex XSIAM (SIEM module)",
    "SOAR": "Cortex XSOAR / XSIAM built-in SOAR",
    "scenariusz": "SOAR Playbook",
    "playbook": "SOAR Playbook",
    "korelacja zdarzeń": "XSIAM analytics rules / correlation rules",
    "reguła korelacji": "XSIAM correlation rule",

    # Threat intelligence
    "threat intelligence": "AutoFocus / Cortex Threat Intel Management",
    "TIP": "Cortex XSIAM TIM / threat intel feeds",

    # Misc
    "ASM": "Cortex Xpanse / XSIAM ASM module",
    "powierzchnia ataku": "Cortex Xpanse",
    "chatbot": "Cortex Copilot",
    "asystent AI": "Cortex Copilot",
    "konsola HTML5": "Cortex management console (browser-based)",

    # English aliases — żeby działało symetrycznie dla EN SIWZ
    "intermediary node": "Broker VM",
    "edr agent": "Cortex XDR Agent",
    "sandbox system": "WildFire",
    "soar playbook": "SOAR Playbook",
    "data lake": "Cortex Data Lake",
    "ai assistant": "Cortex Copilot",
    "attack surface": "Cortex Xpanse",
}


# ═══════════════════════════════════════════════════════════════════════════════
# WARSTWA 3: Synonimy do query expansion
# ═══════════════════════════════════════════════════════════════════════════════

SYNONYMS: Dict[str, List[str]] = {
    "edr": ["Cortex XDR Agent", "endpoint detection response", "agent malware protection"],
    "antywirus": ["Cortex XDR Agent", "malware protection", "EPP"],
    "sandbox": ["WildFire", "dynamic file analysis"],
    "siem": ["Cortex XSIAM", "log analytics", "correlation"],
    "soar": ["Cortex XSOAR", "playbook automation", "incident response"],
    "broker": ["Broker VM", "intermediary node", "on-prem broker"],
    "asm": ["Cortex Xpanse", "attack surface management", "external scanning"],
    "ueba": ["XSIAM analytics", "user entity behavior analytics"],
    "kompatybilność": ["compatibility matrix", "supported operating systems", "OS support"],
    "wsparcie": ["supported", "compatible", "compatibility"],
    "agent windows": ["Cortex XDR Agent Windows", "Windows endpoint agent"],
    "agent linux": ["Cortex XDR Agent Linux", "Linux endpoint agent", "eBPF user-mode kernel module"],
    "agent macos": ["Cortex XDR Agent macOS", "Mac endpoint agent"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Funkcje pomocnicze
# ═══════════════════════════════════════════════════════════════════════════════


def get_architecture_concepts(
    products: Optional[List[str]] = None,
    language: str = "pl",
) -> str:
    """Zwróć złączone koncepcje architektoniczne dla wybranych produktów.

    Używane w system prompt — daje modelowi "kotwicę" co to za produkty.
    """
    if not products:
        products = list(ARCHITECTURE_CONCEPTS.keys())
    lang = language if language in ("pl", "en") else "pl"
    parts: list[str] = []
    seen: set[str] = set()
    for p in products:
        if p not in ARCHITECTURE_CONCEPTS or p in seen:
            continue
        seen.add(p)
        block = ARCHITECTURE_CONCEPTS[p][lang]
        header = f"### {p}" if language == "en" else f"### {p}"
        parts.append(f"{header}\n{block}")
    return "\n\n".join(parts)


def get_term_mappings_block(language: str = "pl") -> str:
    """Wstrzykiwane do prompt jako 'słownik' do tłumaczenia SIWZ ↔ Cortex."""
    lines = []
    for k, v in TERM_MAPPING.items():
        lines.append(f"- '{k}' → {v}")
    label = "TERMINOLOGIA: tłumaczenie pojęć SIWZ na produkt:" if language == "pl" else "TERMINOLOGY: translate RFP concepts to product terms:"
    return label + "\n" + "\n".join(lines)


def expand_query(query: str, max_expansions: int = 4) -> List[str]:
    """Wyszukaj synonimy/aliasy w query i zwróć listę rozszerzonych zapytań."""
    q_lower = query.lower()
    expansions: list[str] = []
    seen: set[str] = set()
    for trigger, expansions_list in SYNONYMS.items():
        if trigger in q_lower:
            for e in expansions_list:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    expansions.append(e)
    return expansions[:max_expansions]


def get_relevant_terms_for_requirement(requirement: str) -> List[str]:
    """Zwróć listę pasujących mapowań termów dla danego wymagania.

    Używane do "podpowiedzi" w prompt'cie — wskazuje LLM jakie tłumaczenia
    pojęć są istotne dla tego konkretnego wymagania.
    """
    r_lower = requirement.lower()
    hits: list[str] = []
    for siwz_term, product_term in TERM_MAPPING.items():
        if siwz_term.lower() in r_lower:
            hits.append(f"'{siwz_term}' → {product_term}")
    return hits[:6]
