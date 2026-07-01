# Traduction Français ↔ Fulfulde (Adamawa, Cameroun) — Projet de faisabilité

Démonstration de faisabilité d'un système de traduction automatique français ↔ fulfulde
(dialecte Adamawa, Cameroun), par fine-tuning de **NLLB-200-distilled-600M** (Meta AI) sur
un corpus parallèle de **4680 versets bibliques alignés**.

## Contexte et objectif

Ce projet est une **preuve de concept**, pas un produit fini. L'objectif : vérifier qu'avec un
volume de données limité mais de bonne qualité (texte biblique aligné phrase à phrase), on peut
obtenir un système de traduction exploitable sur ce registre, avant d'investir dans la collecte
d'un corpus plus large et plus général (conversation courante, administratif, etc.).

## Pourquoi NLLB-200 ?

[NLLB-200](https://ai.meta.com/blog/nllb-200-high-quality-machine-translation/) (Meta AI, 2022)
est un modèle de traduction couvrant nativement 200 langues, avec un accent particulier sur les
langues africaines peu dotées. Il inclut déjà un code de langue fulfulde (`fuv_Latn`), ce qui
permet de **partir d'un modèle qui connaît déjà la structure générale de la langue** plutôt que
d'entraîner un système de zéro — crucial vu la taille réduite de notre corpus.

### ⚠️ Limite importante : dialecte

Le code `fuv_Latn` dans NLLB-200 correspond au **Nigerian Fulfulde**, pas spécifiquement au
fulfulde Adamawa Cameroun. Ce sont des variétés proches mais pas identiques. Ce projet réutilise
ce code comme point de départ et le fine-tuning l'adapte vers le dialecte cible à partir des
données fournies. Une **validation par un locuteur natif de la zone Adamawa** est indispensable
avant toute utilisation au-delà de la démonstration — c'est fait dans ce projet par le porteur
du projet lui-même, locuteur natif basé dans la région.

### Pourquoi le modèle 600M (et pas plus gros) ?

Avec ~4680 paires de phrases, un modèle plus gros (1.3B, 3.3B) augmente le risque de
sur-apprentissage (le modèle mémorise les phrases d'entraînement plutôt que d'apprendre à
généraliser) sans garantie de meilleure qualité. Le 600M distillé offre le meilleur compromis
capacité / risque pour ce volume de données, et tourne confortablement sur un GPU Colab gratuit
(T4).

### Pourquoi bidirectionnel (FR→FF et FF→FR) ?

Chaque paire de phrases génère deux exemples d'entraînement (une direction et l'autre), ce qui
double le signal effectif (~9360 exemples) en partageant les mêmes poids du modèle pour les deux
sens. C'est une forme de régularisation gratuite, particulièrement utile sur petit corpus.

## Structure du dépôt

```
fr-fulfulde-mt/
├── data/
│   ├── raw/              # corpus.csv original (non versionné, voir .gitignore)
│   └── processed/        # corpus nettoyé + train/val/test (générés, non versionnés)
├── notebooks/
│   └── finetune_nllb_colab.ipynb   # pipeline complet, à exécuter sur Google Colab
├── scripts/
│   ├── validate_data.py  # validation non bloquante du corpus (rapport, pas de suppression)
│   └── split_data.py     # split train/val/test reproductible (seed fixe)
├── reports/               # rapports générés (validation, métriques de test)
├── requirements.txt
└── README.md
```

## Méthodologie détaillée

1. **Validation des données** (`scripts/validate_data.py`) : vérifie l'encodage UTF-8, les
   lignes vides, les doublons exacts, les cas où une même phrase française est alignée à deux
   traductions fulfulde différentes, les ratios de longueur suspects (signe possible de
   désalignement), les espaces parasites et les caractères inattendus. **Rien n'est supprimé
   automatiquement** sauf les lignes totalement vides des deux côtés — le reste est à examiner
   manuellement via le rapport généré, le pipeline n'est jamais bloqué par ces avertissements.

2. **Split train/val/test** : 90 % / 5 % / 5 % (~4200 / 235 / 235 paires), seed fixe (42) pour
   la reproductibilité.

3. **Fine-tuning bidirectionnel** de `facebook/nllb-200-distilled-600M` via `Seq2SeqTrainer`
   (Hugging Face Transformers), avec :
   - Checkpointing automatique sur Google Drive toutes les 100 étapes
   - **Reprise automatique** après interruption Colab (détection du dernier checkpoint)
   - Limitation à 3 checkpoints conservés (évite de saturer le Drive)
   - Sélection du meilleur modèle selon la perte de validation (`eval_loss`)

4. **Évaluation** : BLEU et chrF++ sur le jeu de test, jamais vu pendant l'entraînement.
   chrF++ (mesure au niveau caractère) est généralement plus informatif que BLEU pour une
   langue morphologiquement riche et peu dotée comme le fulfulde.

5. **Démo d'inférence** + sauvegarde du modèle final sur Drive.

## Utilisation

### Sur Google Colab (recommandé)

1. Ouvre `notebooks/finetune_nllb_colab.ipynb` dans Google Colab.
2. Active un GPU gratuit : `Exécution > Modifier le type d'exécution > T4 GPU`.
3. Dépose ton `corpus.csv` (colonnes `french_text`, `fulfulde_text`) dans
   `Mon Drive/fr-fulfulde-mt/data/raw/corpus.csv`.
4. Exécute les cellules dans l'ordre. Tout est sauvegardé sur Drive au fur et à mesure.

### En local (validation des données uniquement)

```bash
pip install -r requirements.txt

python scripts/validate_data.py \
    --input data/raw/corpus.csv \
    --output-report reports/validation_report.txt \
    --output-clean data/processed/corpus_clean.csv

python scripts/split_data.py \
    --input data/processed/corpus_clean.csv \
    --outdir data/processed
```

## Résultats

À compléter après exécution : `reports/test_metrics.json` (BLEU, chrF++) sera généré
automatiquement par le notebook. Ajoute ici un tableau récapitulatif une fois l'entraînement
terminé.

| Direction | BLEU | chrF++ |
|---|---|---|
| FR → FF | _à compléter_ | _à compléter_ |
| FF → FR | _à compléter_ | _à compléter_ |

## Limites connues

- **Dialecte** : `fuv_Latn` ≈ Nigerian Fulfulde, pas un code Adamawa Cameroun dédié (voir
  ci-dessus). Validation humaine indispensable.
- **Domaine** : corpus 100 % biblique → registre soutenu/archaïque, vocabulaire religieux.
  Performance non garantie hors de ce domaine.
- **Volume de données** : 4680 paires est petit pour du MT neuronal. Les résultats doivent être
  interprétés comme une preuve de faisabilité, pas comme un système de production.
- **Droits sur le corpus** : si ce corpus provient d'une traduction biblique existante
  (société biblique, mission de traduction, etc.), vérifie les droits de réutilisation/diffusion
  avant de rendre le dépôt public avec les données incluses. Par précaution, `data/raw/` et
  `data/processed/` sont exclus du dépôt Git (voir `.gitignore`) — seuls le code et les rapports
  sont versionnés par défaut.

## Prochaines étapes

Une fois la faisabilité validée sur ce corpus biblique, l'étape suivante consiste à élargir le
corpus avec des données hors domaine (conversation courante, presse, administratif) pour viser
un système de traduction plus généraliste.

## Licence

Code sous licence MIT (voir `LICENSE`). Les données bibliques ne sont **pas couvertes** par
cette licence — vérifie les droits associés à ta source avant toute redistribution.
