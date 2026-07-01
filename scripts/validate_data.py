#!/usr/bin/env python3
"""
validate_data.py
-----------------
Vérifie la qualité d'un corpus parallèle français / fulfulde (CSV à deux
colonnes: french_text, fulfulde_text) SANS rien supprimer automatiquement
(sauf les lignes totalement vides des deux côtés, qui sont inexploitables).

Le script produit un rapport texte listant les anomalies détectées, pour que
l'utilisateur décide lui-même quoi corriger. Il n'interrompt jamais le
pipeline : les avertissements sont informatifs, pas bloquants.

Usage:
    python scripts/validate_data.py --input data/raw/corpus.csv \
        --output-report reports/validation_report.txt \
        --output-clean data/processed/corpus_clean.csv

Options avancées (✅ nouvelles):
    --ratio-max FLOAT   Seuil haut du ratio longueur FF/FR (défaut: 3.0)
                        Au-dessus → paire suspecte (possible mésalignement)
    --ratio-min FLOAT   Seuil bas  du ratio longueur FF/FR (défaut: 0.33)
                        En-dessous → paire suspecte
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["french_text", "fulfulde_text"]

# Caractères "attendus" en français + alphabet latin étendu utilisé pour
# transcrire le fulfulde (incluant les lettres spécifiques courantes :
# ɓ ɗ ƴ ŋ ʼ et leurs variantes capitales). Tout caractère hors de cet
# ensemble (et hors ponctuation/chiffres usuels) est signalé, pas supprimé.
EXPECTED_EXTRA_CHARS = set("ɓɗƴŋʼʔɲÉéèêëàâäîïôöùûüçÀÂÄÎÏÔÖÙÛÜÇ''\"\"…«»")

# Seuils de ratio par défaut (✅ maintenant configurables via --ratio-max / --ratio-min)
DEFAULT_RATIO_MAX = 3.0
DEFAULT_RATIO_MIN = 0.33


def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit(
            f"[ERREUR BLOQUANTE] Le fichier n'est pas en UTF-8 valide : {e}\n"
            "Ré-encode le fichier en UTF-8 avant de continuer "
            "(ex: avec `iconv -f WINDOWS-1252 -t UTF-8 in.csv > out.csv`)."
        )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[ERREUR BLOQUANTE] Colonnes manquantes: {missing}. "
            f"Colonnes trouvées: {list(df.columns)}"
        )
    return df


def is_blank(x) -> bool:
    return pd.isna(x) or str(x).strip() == ""


def check_unexpected_chars(text: str) -> set:
    """Retourne l'ensemble des caractères 'inattendus' dans une chaîne."""
    unexpected = set()
    for ch in text:
        if ch.isalnum() and ch.isascii():
            continue
        if ch in EXPECTED_EXTRA_CHARS:
            continue
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        unexpected.add(ch)
    return unexpected


def run_validation(df: pd.DataFrame,
                   ratio_max: float = DEFAULT_RATIO_MAX,
                   ratio_min: float = DEFAULT_RATIO_MIN) -> dict:
    """
    Args:
        df        : DataFrame corpus
        ratio_max : seuil haut ratio longueur FF/FR — configurable via CLI (✅)
        ratio_min : seuil bas  ratio longueur FF/FR — configurable via CLI (✅)
    """
    report = {
        "n_total": len(df),
        "ratio_max": ratio_max,
        "ratio_min": ratio_min,
        "empty_french": [],
        "empty_fulfulde": [],
        "duplicates_exact": [],
        "duplicates_fr_only": [],
        "length_ratio_outliers": [],
        "unexpected_chars": Counter(),
        "rows_with_unexpected_chars": [],
        "html_or_markup": [],
        "leading_trailing_space": [],
        "rows_to_drop": [],  # uniquement lignes 100% vides des deux côtés
    }

    seen_pairs = {}
    seen_fr = {}

    for idx, row in df.iterrows():
        fr = "" if pd.isna(row["french_text"])   else str(row["french_text"])
        ff = "" if pd.isna(row["fulfulde_text"]) else str(row["fulfulde_text"])

        fr_blank = is_blank(row["french_text"])
        ff_blank = is_blank(row["fulfulde_text"])

        if fr_blank and ff_blank:
            report["rows_to_drop"].append(idx)
            continue
        if fr_blank:
            report["empty_french"].append(idx)
        if ff_blank:
            report["empty_fulfulde"].append(idx)

        if fr != fr.strip() or ff != ff.strip():
            report["leading_trailing_space"].append(idx)

        if re.search(r"<[^>]+>|&[a-z]+;", fr) or re.search(r"<[^>]+>|&[a-z]+;", ff):
            report["html_or_markup"].append(idx)

        if not fr_blank and not ff_blank:
            key = (fr.strip().lower(), ff.strip().lower())
            if key in seen_pairs:
                report["duplicates_exact"].append((seen_pairs[key], idx))
            else:
                seen_pairs[key] = idx

            fr_key = fr.strip().lower()
            if fr_key in seen_fr and seen_fr[fr_key] != idx:
                report["duplicates_fr_only"].append((seen_fr[fr_key], idx))
            else:
                seen_fr[fr_key] = idx

            len_fr, len_ff = len(fr.split()), len(ff.split())
            if len_fr > 0 and len_ff > 0:
                ratio = len_ff / len_fr
                # ✅ Seuils configurables (défaut: 3.0 / 0.33)
                if ratio > ratio_max or ratio < ratio_min:
                    report["length_ratio_outliers"].append(
                        (idx, len_fr, len_ff, round(ratio, 2))
                    )

        for src_text in (fr, ff):
            unexpected = check_unexpected_chars(src_text)
            if unexpected:
                report["unexpected_chars"].update(unexpected)
                report["rows_with_unexpected_chars"].append(idx)

    return report


def write_report(report: dict, out_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("RAPPORT DE VALIDATION - Corpus français / fulfulde")
    lines.append("=" * 70)
    lines.append(f"Lignes totales analysées : {report['n_total']}")
    lines.append(f"Lignes 100% vides (supprimées automatiquement) : {len(report['rows_to_drop'])}")
    lines.append(f"Seuil ratio longueur utilisé : [{report['ratio_min']} ; {report['ratio_max']}]")
    lines.append("")

    def section(title, items, formatter=lambda x: str(x), max_show=20):
        lines.append(f"--- {title} ({len(items)}) ---")
        if not items:
            lines.append("  Aucun problème détecté.")
        else:
            for it in items[:max_show]:
                lines.append(f"  {formatter(it)}")
            if len(items) > max_show:
                lines.append(f"  ... et {len(items) - max_show} de plus (voir CSV si besoin).")
        lines.append("")

    section("Lignes avec texte français vide (mais fulfulde présent)",
             report["empty_french"], lambda i: f"ligne {i}")
    section("Lignes avec texte fulfulde vide (mais français présent)",
             report["empty_fulfulde"], lambda i: f"ligne {i}")
    section("Paires dupliquées exactes (fr+ff identiques)",
             report["duplicates_exact"], lambda p: f"lignes {p[0]} et {p[1]}")
    section("Même phrase française alignée à 2 fulfulde différents",
             report["duplicates_fr_only"], lambda p: f"lignes {p[0]} et {p[1]} (vérifier cohérence)")
    section(
        f"Ratio de longueur fr/ff suspect (> {report['ratio_max']} ou < {report['ratio_min']})",
        report["length_ratio_outliers"],
        lambda t: f"ligne {t[0]}: {t[1]} mots FR vs {t[2]} mots FF (ratio={t[3]})"
    )
    section("Espaces en début/fin de champ",
             report["leading_trailing_space"], lambda i: f"ligne {i}")
    section("Balises HTML ou entités résiduelles",
             report["html_or_markup"], lambda i: f"ligne {i}")
    section("Lignes contenant des caractères inattendus",
             report["rows_with_unexpected_chars"], lambda i: f"ligne {i}")

    if report["unexpected_chars"]:
        lines.append("--- Caractères inattendus rencontrés (fréquence) ---")
        for ch, count in report["unexpected_chars"].most_common(30):
            display = ch if ch.isprintable() else f"U+{ord(ch):04X}"
            lines.append(f"  '{display}' : {count} occurrence(s)")
        lines.append("")
        lines.append("  NB: certains de ces caractères sont peut-être des lettres")
        lines.append("  fulfulde légitimes non listées dans EXPECTED_EXTRA_CHARS.")
        lines.append("  Ajoute-les au script si c'est le cas pour éviter les faux positifs.")
        lines.append("")

    lines.append("=" * 70)
    lines.append("RAPPEL : ce script n'a rien supprimé sauf les lignes totalement vides.")
    lines.append("Toutes les autres anomalies sont à examiner manuellement avant entraînement.")
    lines.append("=" * 70)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport écrit dans : {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",         required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--output-clean",  required=True, type=Path,
                        help="CSV de sortie (lignes 100% vides retirées uniquement)")
    # ✅ Nouveaux arguments : seuils de ratio configurables
    parser.add_argument("--ratio-max", type=float, default=DEFAULT_RATIO_MAX,
                        help=f"Seuil haut ratio longueur FF/FR (défaut: {DEFAULT_RATIO_MAX}). "
                             "Au-dessus → paire signalée comme suspecte.")
    parser.add_argument("--ratio-min", type=float, default=DEFAULT_RATIO_MIN,
                        help=f"Seuil bas ratio longueur FF/FR (défaut: {DEFAULT_RATIO_MIN}). "
                             "En-dessous → paire signalée comme suspecte.")
    args = parser.parse_args()

    if args.ratio_min <= 0 or args.ratio_max <= 0 or args.ratio_min >= args.ratio_max:
        raise SystemExit(
            f"[ERREUR] Les seuils de ratio doivent vérifier 0 < ratio_min < ratio_max. "
            f"Valeurs reçues : min={args.ratio_min}, max={args.ratio_max}"
        )

    df = load_csv(args.input)
    report = run_validation(df, ratio_max=args.ratio_max, ratio_min=args.ratio_min)
    write_report(report, args.output_report)

    df_clean = df.drop(index=report["rows_to_drop"]).reset_index(drop=True)
    args.output_clean.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.output_clean, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    print(f"Corpus nettoyé écrit dans : {args.output_clean} ({len(df_clean)} lignes)")

    n_warnings = (
        len(report["empty_french"]) + len(report["empty_fulfulde"])
        + len(report["duplicates_exact"]) + len(report["duplicates_fr_only"])
        + len(report["length_ratio_outliers"]) + len(report["rows_with_unexpected_chars"])
    )
    if n_warnings > 0:
        print(f"\n⚠️  {n_warnings} avertissement(s) au total — consulte le rapport. "
              "Le pipeline continue normalement (non bloquant).")
    else:
        print("\n✅ Aucune anomalie détectée.")


if __name__ == "__main__":
    main()
