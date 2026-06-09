# Implémentation et Fine-tuning d'un Large Language Model

> Projet de semestre 8 — Polytech Lyon, Mathématiques Appliquées et Modélisation (MAM 4A)  
> Étudiant : **Mouhamed Bachir CISSE** | Encadrant : **HONORE IGOR**

## Objectif

Ce projet explore les mécanismes fondamentaux des Large Language Models (LLMs) en deux volets complémentaires :

1. **Implémentation from scratch** d'un mini-Transformer en Python/PyTorch
2. **Fine-tuning** du modèle open-source Mistral-7B avec la technique LoRA sur le dataset GSM8K

L'objectif principal n'est pas de rivaliser avec les LLMs industriels, mais de **comprendre et maîtriser l'architecture Transformer de l'intérieur** en mobilisant les outils mathématiques du cursus MAM.

---

## Structure du projet

```
├── transformers.ipynb      # Mini-Transformer from scratch (VS Code)
├── fine_tuning.ipynb       # Fine-tuning Mistral-7B avec LoRA (Kaggle)
├── loss_mini_transformer.png
├── loss_fine_tuning.png
├── comparaison_parametres.png
└── README.md
```

---

## Volet 1 — Mini-Transformer from scratch

Implémentation complète d'un Transformer en Python/PyTorch, brique par brique :

| Classe | Rôle |
|--------|------|
| `Embedding` | Conversion tokens → vecteurs (dim=512) |
| `PositionalEncoding` | Encodage sinusoïdal de la position |
| `MultiHeadAttention` | Attention multi-têtes (h=8) |
| `FeedForward` | Réseau feed-forward (ReLU, dim=2048) |
| `TransformerBlock` | Bloc complet avec connexions résiduelles |
| `MiniTransformer` | Modèle final (N=6 blocs) |

**Résultats :**
- 51,702,016 paramètres entraînables
- Loss : 10.46 → 0.02 en 200 époques
- Validation de l'architecture et de la rétropropagation ✓

---

## Volet 2 — Fine-tuning Mistral-7B avec LoRA

Fine-tuning du modèle [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) sur le dataset [GSM8K](https://huggingface.co/datasets/openai/gsm8k) via la technique **LoRA (Low-Rank Adaptation)**.

### Configuration LoRA

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Hyperparamètres d'entraînement

| Paramètre | Valeur |
|-----------|--------|
| Steps | 200 |
| Batch size | 2 |
| Learning rate | 2×10⁻⁴ |
| Rang LoRA (r) | 8 |
| Précision | BF16 |
| GPU | NVIDIA T4 (Kaggle) |

### Résultats

- Paramètres entraînés : **6,815,744** (0.094% du total)
- Loss : 1.025 → 0.722 (−30%) en 200 steps
- Validation du pipeline LoRA ✓

**Modèle fine-tuné disponible sur Hugging Face :**  
🤗 [bachir6c/mistral-mam-lora](https://huggingface.co/bachir6c/mistral-mam-lora)

---

## Stack technique

- Python 3.11
- PyTorch
- Hugging Face (transformers, peft, trl, datasets)
- Kaggle (GPU T4)

---

## Limites et perspectives

- GSM8K est un dataset d'arithmétique primaire en anglais — inadapté pour des maths universitaires en français
- 200 steps représentent seulement 5% du dataset
- **Perspectives :** dataset MAM en français, 1000+ steps, interface Gradio/Streamlit

---

## Référence

- Vaswani et al. (2017) — *Attention is All You Need*
- Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
- Jiang et al. (2023) — *Mistral 7B*
