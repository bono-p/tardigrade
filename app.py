"""
app.py — Interface Gradio pour le modèle de traduction Français ↔ Fulfulde Adamawa.
À déployer tel quel sur Hugging Face Spaces (SDK Gradio).

Prérequis :
  - Le modèle est publié sur le Hub sous : bonopassale/nllb-fra-fuv-finetuned
  - requirements.txt contient : transformers, sentencepiece, gradio, torch
"""

import torch
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Configuration ─────────────────────────────────────────────────────────────
HUB_REPO_ID = "bonopassale/nllb-fra-fuv-finetuned"
SRC_LANG    = "fra_Latn"   # Français
TGT_LANG    = "fuv_Latn"   # Fulfulde (code NLLB le plus proche : Nigerian Fulfulde)
MAX_LENGTH  = 192          # Aligné avec le fine-tuning

# ── Chargement du modèle ──────────────────────────────────────────────────────
print(f"Chargement du modèle {HUB_REPO_ID}…")
tokenizer = AutoTokenizer.from_pretrained(HUB_REPO_ID)
model     = AutoModelForSeq2SeqLM.from_pretrained(HUB_REPO_ID)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Modèle chargé sur {device}.")


# ── Fonction de traduction ────────────────────────────────────────────────────
def translate(text: str, direction: str, num_beams: int = 5) -> str:
    """
    Traduit `text` dans la direction choisie.

    Args:
        text      : texte à traduire
        direction : "Français → Fulfulde" ou "Fulfulde → Français"
        num_beams : nombre de faisceaux pour le beam search (1 = greedy, plus rapide)
    Returns:
        Traduction sous forme de chaîne.
    """
    if not text.strip():
        return ""

    if direction == "Français → Fulfulde":
        src, tgt = SRC_LANG, TGT_LANG
    else:
        src, tgt = TGT_LANG, SRC_LANG

    tokenizer.src_lang = src
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),
            num_beams=num_beams,
            max_length=MAX_LENGTH,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


# ── Interface Gradio ──────────────────────────────────────────────────────────
DESCRIPTION = """
## Traduction Français ↔ Fulfulde Adamawa (Cameroun)

Modèle [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
fine-tuné sur ~12 000 paires de phrases parallèles (corpus biblique).

> ⚠️ **Limites** : le modèle est spécialisé sur un registre biblique/soutenu.
> Le code de langue utilisé (`fuv_Latn`) correspond au Fulfulde nigérian dans NLLB-200 ;
> le fine-tuning l'adapte vers le dialecte Adamawa Cameroun.
> Validation humaine recommandée pour tout usage au-delà de la démonstration.
"""

EXAMPLES = [
    ["Au commencement, Dieu créa les cieux et la terre.", "Français → Fulfulde"],
    ["Aimez-vous les uns les autres.", "Français → Fulfulde"],
    ["Ne crains pas, car je suis avec toi.", "Français → Fulfulde"],
    ["Alla woni e mum.", "Fulfulde → Français"],
    ["Ndaa inɗe ɓiɓɓe Yaakubu.", "Fulfulde → Français"],
    ["Yusufu e deerɗiraaɓe muuɗum fuu maayi.", "Fulfulde → Français"],
]

with gr.Blocks(title="FR ↔ Fulfulde Adamawa — DevLab") as iface:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=4,
                label="Texte source",
                placeholder="Saisissez le texte à traduire…",
            )
            direction = gr.Dropdown(
                choices=["Français → Fulfulde", "Fulfulde → Français"],
                value="Français → Fulfulde",
                label="Direction",
            )
            num_beams = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Faisceaux (beam search) — plus élevé = meilleure qualité, plus lent",
            )
            btn = gr.Button("Traduire", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(lines=4, label="Traduction")

    btn.click(fn=translate, inputs=[input_text, direction, num_beams], outputs=output_text)
    input_text.submit(fn=translate, inputs=[input_text, direction, num_beams], outputs=output_text)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, direction],
        label="Exemples prédéfinis",
    )

if __name__ == "__main__":
    iface.launch()
