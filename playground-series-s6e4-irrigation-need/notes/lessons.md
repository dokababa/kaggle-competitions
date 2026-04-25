# Lessons from PS S6E4 that should transfer

## Target encoding choices

- **Multiclass TE on a clean source crushes scalar TE.** Encoding `P(class=c | feature=x)` for each class as 3 separate columns is 3× more informative than encoding the mean of an ordinal-coded target. On this comp it bought +0.0013 LB by itself.
- **Don't compute TE on a noisy training set when a clean small reference set exists.** Ordered TE on the 630K noisy train (R46) actively hurt CV; TE on the 10K clean original was strictly better.

## Threshold optimisation for balanced accuracy

- For class-imbalanced multiclass with balanced-accuracy as the metric, **threshold weights are not optional.** Skipping them collapsed our LB by ~0.007.
- Threshold weights overfit OOF in subtle ways. **Single-seed models thresholded once beat multi-seed averages thresholded once**, even though the multi-seed OOF was higher. This is non-obvious and burned us twice.

## Things that didn't work that I won't try first next time

- **Pseudo-labelling at high keep-rate.** If your model is already strong (LB > 0.97), the pseudo-labels are mostly your own predictions. You're self-distilling, not adding signal.
- **Same-feature-set diversity attempts (LGBM blended with XGB).** When two algorithms see the same features and target, on tabular data with strong feature engineering, their predictions converge. Real diversity needs different inputs or different architectures (NN, kNN).
- **Specialist binary models for the rare class.** Useful when the multiclass model isn't using class weights. Useless when it already is.

## Things to try first next time

- **Identify the data-generating process.** Spend the first day looking at feature distributions: orig vs train vs test. Histogram the digit positions. Snap to grids. The noise structure tells you what FE will and won't help.
- **Encode the rule, not the prediction.** When a deterministic rule exists on clean data, build features that let the model see how each row relates to the rule cut-points (rounding, digit decomposition, distance-to-cut), not just the rule output.
- **Build the threshold-opt and CV harness on Day 1.** It's the framework you'll use for every later experiment, and post-processing decisions interact strongly with model choice.

## Process

- Maintain a single `results.tsv` log with `round | OOF | LB | features | runtime | notes` from round 1. Without it you forget what you tried.
- Set a 6-hour minimum gap between submissions and only submit when CV improves. It forces you to ask "is this actually better" before burning a slot.
