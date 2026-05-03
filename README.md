# kaggle-competitions

I will dump all my kaggle competitions here. One folder per competition, each one a self-contained writeup of how I approached it, what worked, what didn't, the scripts I'd want to reach for again, and the final rank.

I started this because I kept losing the threads of what I learned between competitions and local folders kept getting piled up, scripts got forgotten and the lessons unlearnt. This repo is intended to dump all important competition stuff before deleting locally. 

## Competitions

| # | Competition | Dates | Best Public LB | Final Rank |
|---|-------------|-------|----------------|------------|
| 1 | [Playground Series S6E4 — Predicting Irrigation Need](./playground-series-s6e4-irrigation-need) | Apr 2026 | **0.97785** (balanced accuracy) | 775/4315 |

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

The READMEs are self explanatory for each competition, just open the folder and follow. 

## About me

Data science is an art, and here I am trying to be an artist. 
My kaggle account: https://www.kaggle.com/dokababaa
