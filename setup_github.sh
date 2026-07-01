#!/usr/bin/env bash
# setup_github.sh
# ----------------
# Initialise le dépôt Git localement et pousse vers GitHub.
# À exécuter UNE SEULE FOIS depuis la racine du dépôt (fr-fulfulde-mt/).
#
# Prérequis :
#   - Git installé
#   - Un nouveau dépôt vide créé sur GitHub (https://github.com/new)
#     Nom suggéré : fr-fulfulde-mt
#   - Un token GitHub valide (Personal Access Token, scope "repo")
#     ⚠️  Ne pas mettre le token dans ce fichier — passe-le en variable d'environnement.
#
# Usage :
#   export GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
#   export GITHUB_USER=ton-username
#   bash setup_github.sh

set -e

if [ -z "$GITHUB_USER" ] || [ -z "$GITHUB_TOKEN" ]; then
  echo "❌ Configure d'abord les variables d'environnement :"
  echo "   export GITHUB_USER=ton-username"
  echo "   export GITHUB_TOKEN=ghp_xxxxxxxxxxxxx"
  exit 1
fi

REPO_NAME="fr-fulfulde-mt"
REMOTE_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

git init
git add .
git commit -m "Initial commit : pipeline fine-tuning NLLB-200 français-fulfulde Adamawa"
git branch -M main
git remote add origin "$REMOTE_URL"
git push -u origin main

echo ""
echo "✅ Dépôt poussé avec succès !"
echo "   https://github.com/${GITHUB_USER}/${REPO_NAME}"
