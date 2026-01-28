# ML Paper Journey 📚

A personal learning tracker for essential Machine Learning and AI papers. Track your reading progress through foundational papers in deep learning, from AlexNet to GPT-4.

![Preview](preview.png)

## Features

- 🎨 **Bento Grid Layout** - Modern Apple-style card design
- 🌙 **Read/Unread States** - Unread papers appear grayscale, read papers glow
- 🏷️ **Category Filtering** - Filter by Foundations, Language Models, Multimodal, Efficiency, or Data
- 📝 **Reading Notes** - Track your thoughts with markdown files
- 📊 **Progress Tracking** - Visual progress ring shows completion percentage
- 📱 **Responsive** - Works on desktop, tablet, and mobile

## Quick Start

### 1. Deploy to GitHub Pages

1. Fork this repository
2. Go to Settings → Pages
3. Set source to "Deploy from a branch"
4. Select `main` branch and `/docs` folder
5. Your site will be live at `https://yourusername.github.io/repo-name/`

### 2. Track Your Reading

Papers start as **unread** (grayscale cards). To mark a paper as **read**:

1. Create a markdown file in `docs/notes/` with the paper's ID
2. Add your reading notes
3. The card will automatically light up!

**Example**: To mark "Attention Is All You Need" as read:

```bash
# Create the notes file
touch docs/notes/transformer-2017.md
```

Then add your notes:

```markdown
# Attention Is All You Need - Notes

## Key Takeaways
- Self-attention replaces recurrence
- Multi-head attention allows parallel processing
- Positional encoding preserves sequence order

## Questions
- How does the attention mechanism scale?
```

### 3. Paper IDs Reference

| Category | Paper | ID |
|----------|-------|-----|
| 🏛️ Foundations | Brook for GPUs | `brook-gpu-2004` |
| 🏛️ Foundations | AlexNet | `alexnet-2012` |
| 🏛️ Foundations | GAN | `gan-2014` |
| 🏛️ Foundations | ResNet | `resnet-2015` |
| 🏛️ Foundations | Transformer | `transformer-2017` |
| 💬 Language Models | Word2Vec | `word2vec-2013` |
| 💬 Language Models | Seq2Seq | `seq2seq-2014` |
| 💬 Language Models | Bahdanau Attention | `bahdanau-attention-2015` |
| 💬 Language Models | GNMT | `gnmt-2016` |
| 💬 Language Models | GPT-1 | `gpt1-2018` |
| 💬 Language Models | BERT | `bert-2018` |
| 💬 Language Models | GPT-2 | `gpt2-2019` |
| 💬 Language Models | GPT-3 | `gpt3-2020` |
| 💬 Language Models | InstructGPT | `instructgpt-2022` |
| 💬 Language Models | Tülu 3 | `tulu3-2024` |
| 🎨 Multimodal | Two-Stream CNN | `two-stream-2014` |
| 🎨 Multimodal | Video CNN | `video-cnn-2014` |
| 🎨 Multimodal | Diffusion (Thermodynamics) | `diffusion-thermo-2015` |
| 🎨 Multimodal | AlphaGo Zero | `alphago-zero-2017` |
| 🎨 Multimodal | DDPM | `ddpm-2020` |
| 🎨 Multimodal | ViT | `vit-2020` |
| 🎨 Multimodal | CLIP | `clip-2021` |
| 🎨 Multimodal | Latent Diffusion | `latent-diffusion-2021` |
| 🎨 Multimodal | Chain-of-Thought | `cot-2022` |
| 🎨 Multimodal | DiT | `dit-2022` |
| ⚡ Efficiency | Knowledge Distillation | `distillation-2015` |
| ⚡ Efficiency | MoE | `moe-2017` |
| ⚡ Efficiency | ZeRO | `zero-2019` |
| ⚡ Efficiency | Scaling Laws | `scaling-laws-2020` |
| ⚡ Efficiency | LoRA | `lora-2021` |
| ⚡ Efficiency | Chinchilla | `chinchilla-2022` |
| ⚡ Efficiency | ReAct | `react-2022` |
| 📊 Data & Scaling | The Bitter Lesson | `bitter-lesson-2019` |
| 📊 Data & Scaling | LAION-5B | `laion5b-2022` |
| 📊 Data & Scaling | RefinedWeb | `refinedweb-2023` |
| 📊 Data & Scaling | MegaScale | `megascale-2024` |

> 📖 **Paper list source**: [Awesome-AITools](https://github.com/ikaijua/Awesome-AITools) • 美團光年之外產品負責人 謝青池：《AI演義，36篇論文開啟你的探索之旅》

## Project Structure

```
ML_AI/
├── .gitignore              # Git ignore rules
└── docs/                   # GitHub Pages root
    ├── index.html          # Main page
    ├── styles.css          # Styling
    ├── app.js              # JavaScript logic
    ├── papers.json         # Paper metadata
    ├── preview.png         # Preview image
    ├── README.md           # This file
    └── notes/              # Your reading notes
```

## Customization

### Adding New Papers

Edit `docs/papers.json`:

```json
{
  "id": "my-paper-2024",
  "title": "My Paper Title",
  "authors": ["Author 1", "Author 2"],
  "year": 2024,
  "category": "language-models",
  "filename": "paper.pdf",
  "arxiv": "2401.12345",
  "significance": "Brief description of importance",
  "tags": ["Tag1", "Tag2"]
}
```

### Categories

- `foundations` - 🏛️ Core architectures (CNN, Transformer, GAN)
- `language-models` - 💬 NLP & LLMs (GPT, BERT, Word2Vec)
- `multimodal` - 🎨 Vision-Language (CLIP, Diffusion, ViT)
- `efficiency` - ⚡ Optimization (LoRA, MoE, Scaling Laws)
- `data` - 📊 Datasets & Infrastructure (LAION, Scaling)

## Tech Stack

- Pure HTML/CSS/JS (no build step required)
- GitHub Pages compatible
- Markdown support via marked.js (optional)

## License

MIT - Use freely for your personal learning journey!

---

Happy reading! 🚀
