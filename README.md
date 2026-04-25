# kaggle-competitions

A scrapbook of every Kaggle competition I take a serious crack at. One folder per competition, each one a self-contained writeup of how I attacked it, what worked, what didn't, the scripts I'd want to reach for again, and the final rank.

I started this because I kept losing the threads of what I learned between competitions — local folders piled up, scripts got stale, the lessons faded. This repo is the canonical home now: I work locally during a competition, then dump the cleaned-up artifacts here and wipe the local folder. Future-me only needs to look in one place.

## Competitions

| # | Competition | Dates | Best Public LB | Final Rank |
|---|-------------|-------|----------------|------------|
| 1 | [Playground Series S6E4 — Predicting Irrigation Need](./playground-series-s6e4-irrigation-need) | Apr 2026 | **0.97785** (balanced accuracy) | _TBD_ |

## How to read each folder

Each competition folder follows the same shape:

```
<competition-name>/
├── README.md              ← The writeup. Start here.
├── notes/                 ← Lessons, feature inventories, anything worth keeping
├── src/                   ← The scripts that actually mattered (not every dead-end)
├── data/.gitkeep          ← Drop competition data here to reproduce
├── output/.gitkeep        ← Where runtime artifacts land
├── requirements.txt
└── .gitignore
```

The READMEs lean long because the interesting stuff is usually in why something didn't work, not the final pipeline. I write them mostly for myself six months later.

## About me

Data scientist by trade, Kaggle player by obsession. I learn faster by doing things badly first, so most of my competition runs include 30+ rounds of experiments before the actual solution emerges. The repo captures the path, not just the destination.

If you're reading this from a future competition — hi. Keep going.
