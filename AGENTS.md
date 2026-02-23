# AGENTS.md — ML Paper Journey

## Project Purpose
Static GitHub Pages site tracking reading progress through 36 foundational ML/AI papers.
Owner reads papers, writes notes, site auto-detects completion.

## Architecture
- Pure HTML/CSS/JS — NO frameworks, NO build step, NO package.json
- Deployed from `docs/` folder on `main` branch via GitHub Pages
- Single external dependency: `marked.js` (CDN in index.html)
- Remote: `https://github.com/geomingical/ai-paper-tracker.git`

### Data Flow
1. `app.js` fetches `papers.json` (36 papers, 5 categories)
2. For each paper, fetches `notes/{paper.id}.md` via Promise.all
3. File exists + non-empty → paper state = **Read** (colored glow border)
4. File missing/empty → paper state = **Unread** (grayscale + 50% opacity)

Read state is FILE-DRIVEN. No database, no checkbox. Create/delete `.md` file = toggle state.

## Key Files
```
docs/index.html    — Entry point. Cache-busts assets via ?v= query strings
docs/app.js        — Core logic: fetch, filter, render, modal (212 lines)
docs/styles.css    — All styling incl. card glow, grayscale states (1003 lines)
docs/papers.json   — Paper metadata: id, title, authors, year, category, tags
docs/notes/*.md    — Reading notes. Filename = paper id. Existence = read
閱讀順序.md         — Curated 31-paper reading sequence (separate from papers.json's 36)
```

## papers.json Schema
```json
{
  "id": "transformer-2017",       // Used as notes filename + unique key
  "title": "Attention Is All You Need",
  "authors": ["Vaswani", ...],     // UI shows first 3 + "+N"
  "year": 2017,                    // Grid sorted ascending by year
  "category": "foundations",       // foundations|language-models|multimodal|efficiency|data
  "filename": "1706.03762v7.pdf", // PDF in gitignored category folders
  "arxiv": "1706.03762",          // Nullable
  "significance": "...",
  "tags": ["Transformer", ...]     // UI shows first 3
}
```

## Note Format Convention (STRICT)
Filename: `docs/notes/{paper-id}.md` — must match `id` field in papers.json exactly.

```markdown
# Reading Notes

---

# Learning From AI (GPT)      ← or (GPT5.2), model name varies

### 1️⃣ Section Title
Content with `$inline math$` and:
$$
block math
$$

### 2️⃣ Next Section
- 重點 (Key Point) callouts
- 實戰意義 (Practical Significance) callouts
- 👉 bullet highlights

**ChatGPT 對話連結**: [URL]    ← always last line
```

Math: `$$...$$` block, `$...$` inline. NEVER bare brackets `[...]` for math.

## Adding a Paper
1. Add entry to `docs/papers.json` following schema above
2. Place PDF in appropriate Chinese-named category folder (gitignored)
3. To mark as read: create `docs/notes/{id}.md` following note format

## Commit Convention
```
Docs: add {PaperName} ({Year}) reading notes with formatted markdown structure
```

## Gotchas
- PDF folders (基礎模型/, 語言模型/, etc.) and `*.pdf` are gitignored — never commit PDFs
- `index.html` uses `?v=` cache busting on CSS/JS — bump version after changes
- Card rendering slices authors to 3 and tags to 3 — data can have more
- Category colors defined as `--card-accent` CSS variable per category
- Notes are rendered via `marked.parse()` in a modal — test markdown renders correctly
- 閱讀順序.md has 31 papers; papers.json has 36 (5 foundations already read at project start)