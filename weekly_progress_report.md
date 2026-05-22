# Weekly Progress Report

**Date:** May 14, 2026  
**Project:** Summer Twin / Hemodynamic Agent Baseline

## Summary

This week, I built a minimal baseline for a future agent tool that predicts missing hemodynamic summaries from partial user-provided summary values. The baseline follows a maximum-likelihood-estimation-inspired idea and is intended to serve as the first functional prototype for future optimization.

## Completed Work

- Built a simple local UI for entering known hemodynamic summary values.
- Added a table-based input interface where users can provide one or more values, such as `SBP = 100` and `MAP = 90`.
- Added a second table where users can select which hemodynamic summaries they want to predict, such as `sPAP` and `dPAP`.
- Connected the UI to a backend prediction workflow.
- Implemented the first version of the prediction baseline using existing CSV data with 40,000 simulated cases.
- Verified an example query using `SBP = 100` and `MAP = 90` with a tolerance of +/- 1, which returned 25 matched cases.
- Extracted predicted target summaries such as `sPAP` and `dPAP` from the matched or nearby cases.

## Method

The current baseline uses a maximum-likelihood-estimation-inspired strategy. Since the number of requested predicted summaries may vary, the method uses different strategies depending on the output dimension:

- For 2-4 predicted summaries: conditional KDE over real samples.
- For 4-10 predicted summaries: weighted neighbors with clustering/medoids.
- For 10+ predicted summaries: dimensionality reduction followed by clustering, then mapping back to real cases.

## Data

The data comes from a CSV file with approximately 40,000 cases. Each row represents one case:

- Column A: `case_id`
- Columns B-N: descriptive cardiovascular model parameters
- Columns O-BB: hemodynamic summaries, such as `SBP`, `DBP`, `MAP`, and `PP`

## Current Status

The current prototype is functional as a minimal baseline. It supports structured numeric inputs, selectable prediction targets, and a simple prediction workflow connected to the UI.

## Next Steps

- Improve the maximum-likelihood-based prediction algorithm.
- Refine the strategy for high-dimensional summary prediction.
- Design a query parser that can convert user requests into structured fields.
- Prepare the baseline to become a reusable agent tool.
