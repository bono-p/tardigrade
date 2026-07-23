# Traduction Français ↔ Fulfulde (Adamawa, Cameroun) — Projet de faisabilité

Démonstration de faisabilité d'un système de traduction automatique français ↔ fulfulde
(dialecte Adamawa, Cameroun), par fine-tuning de **NLLB-200-distilled-600M** (Meta AI) sur
un corpus parallèle de **12 000 paires de phrases alignées**.

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
avant toute utilisation au-delà de la démonstration.

### Pourquoi le modèle 600M (et pas plus gros) ?

Avec ~12 000 paires de phrases, un modèle plus gros (1.3B, 3.3B) reste envisageable mais risque
de sur-apprendre sur un corpus à dominante biblique. Le 600M distillé offre le meilleur
compromis capacité / risque sur GPU Colab gratuit (T4). Si le corpus dépasse les 30 000 paires
ou sort du domaine biblique, le passage au 1.3B vaut la peine d'être testé.

### Pourquoi bidirectionnel (FR→FF et FF→FR) ?

Chaque paire de phrases génère deux exemples d'entraînement (une direction et l'autre), ce qui
porte le signal effectif à ~21 600 exemples avec 12 000 paires, en partageant les mêmes poids
du modèle pour les deux sens. C'est une forme de régularisation gratuite, particulièrement
utile sur corpus de taille modeste.

> **Note sur la qualité par direction** : le français étant mieux représenté dans les données
> de pré-entraînement de NLLB-200, la direction FR→FF bénéficie d'un meilleur encodage de
> départ. La direction FF→FR améliore à mesure que le fine-tuning progresse et que le modèle
> apprend les représentations du fulfulde Adamawa.

## Structure du dépôt

```
tardigrade/
├── app.py                # Interface Gradio — déploiement Hugging Face Spaces
├── data/
│   ├── raw/              # corpus.csv original (non versionné, voir .gitignore)
│   └── processed/        # corpus nettoyé + train/val/test (générés, non versionnés)
├── notebooks/
│   └── finetune_nllb_colab.ipynb   # pipeline complet, à exécuter sur Google Colab
├── scripts/
│   ├── validate_data.py  # validation + normalisation NFC du corpus
│   └── split_data.py     # split train/val/test reproductible (seed fixe)
├── reports/              # rapports générés (validation, métriques de test)
├── requirements.txt
└── README.md
```

## Méthodologie détaillée

1. **Validation et normalisation des données** (`scripts/validate_data.py`) : vérifie l'encodage
   UTF-8, les lignes vides, les doublons exacts, les ratios de longueur suspects, les espaces
   parasites et les caractères inattendus. **Applique une normalisation Unicode NFC** sur les deux
   colonnes (activée par défaut) pour résoudre les incohérences d'encodage des caractères fulfulde
   (`ɓ`, `ɗ`, `ƴ`, `ŋ`…) selon la source du corpus. Rien n'est supprimé automatiquement sauf
   les lignes vides des deux côtés.

2. **Split train/val/test** : 90 % / 5 % / 5 % (~10 800 / 600 / 600 paires), seed fixe (42)
   pour la reproductibilité.

3. **Fine-tuning bidirectionnel** de `facebook/nllb-200-distilled-600M` via `Seq2SeqTrainer`
   (Hugging Face Transformers), avec :
   - Normalisation NFC inline au chargement du corpus
   - Checkpointing automatique sur Google Drive toutes les 500 étapes
   - **Reprise automatique** après interruption Colab (détection du dernier checkpoint)
   - Datasets de validation par direction (`val_fr2ff`, `val_ff2fr`) pour diagnostic post-entraînement
   - Sélection du meilleur modèle selon la perte de validation (`eval_loss`)

4. **Évaluation** : BLEU et chrF++ séparément pour FR→FF et FF→FR sur le jeu de test.

5. **Démo d'inférence** bidirectionnelle (FR→FF et FF→FR) + sauvegarde sur Drive.

6. **Interface Gradio** (`app.py`) : prête à déployer sur Hugging Face Spaces une fois le
   modèle publié sous `bonopassale/nllb-fra-fuv-finetuned`.

## Utilisation

### Sur Google Colab (recommandé)

1. Ouvre `notebooks/finetune_nllb_colab.ipynb` dans Google Colab.
2. Active un GPU : `Exécution > Modifier le type d'exécution > T4 GPU`.
3. Dépose ton `corpus.csv` (colonnes `french_text`, `fulfulde_text`) dans
   `Mon Drive/fr-fulfulde-mt/data/raw/corpus.csv`.
4. Exécute les cellules dans l'ordre. Tout est sauvegardé sur Drive au fur et à mesure.

### En local (validation des données)

```bash
pip install -r requirements.txt

# Validation + normalisation NFC (activée par défaut)
python scripts/validate_data.py \
    --input data/raw/corpus.csv \
    --output-report reports/validation_report.txt \
    --output-clean data/processed/corpus_clean.csv

python scripts/split_data.py \
    --input data/processed/corpus_clean.csv \
    --outdir data/processed
```

### Interface Gradio (local ou Spaces)

```bash
pip install gradio
python app.py
```

## Résultats

À compléter après exécution : `reports/test_metrics.json` (BLEU, chrF++) sera généré
automatiquement par le notebook.

| Direction | BLEU | chrF++ |
|---|---|---|
| FR → FF | _à compléter_ | _à compléter_ |
| FF → FR | _à compléter_ | _à compléter_ |

## Limites connues

- **Dialecte** : `fuv_Latn` ≈ Nigerian Fulfulde. Validation humaine indispensable.
- **Domaine** : corpus à dominante biblique → registre soutenu. Performance non garantie hors
  de ce registre.
- **Volume** : 12 000 paires est un corpus de taille modeste pour du MT neuronal. Les résultats
  doivent être interprétés comme une preuve de faisabilité.
- **Droits sur le corpus** : si le corpus provient d'une traduction biblique existante,
  vérifie les droits avant de rendre le dépôt public avec les données. `data/` est exclu du
  dépôt Git par `.gitignore`.

## Prochaines étapes

- Élargir le corpus avec des données hors domaine (conversation, presse, administratif).
- Tester le modèle 1.3B si le corpus dépasse 30 000 paires.
- Évaluation humaine par des locuteurs natifs Adamawa.
- Surveiller la qualité FF→FR indépendamment (datasets `val_fr2ff` / `val_ff2fr`).

## Licence

Code sous licence MIT (voir `LICENSE`). Les données ne sont **pas couvertes** par cette
licence — vérifie les droits associés à ta source avant toute redistribution.
