# Tool Wear Prediction in Engine Manufacturing

## Project Goal

The goal of this project is to analyze and model tooling wear and tear during engine part manufacturing. Using sensor data and usage logs, we want to predict the condition of cutting tools — whether they’re still usable, worn out, or close to failure — and potentially estimate how much life they have left.

## Why This Matters

Tool wear directly affects part quality, machine downtime, and production costs. By predicting wear before failure, manufacturers can:
- Replace tools at the right time (not too early, not too late)
- Avoid unexpected breakdowns or poor-quality parts
- Reduce tooling costs and downtime
- Move toward smarter, predictive maintenance workflows

## Problem Type

This could be approached in a few different ways:
- **Classification**: Label tools as Healthy / Worn / Broken
- **Regression**: Predict the amount of wear or remaining tool life
- **Time Series**: Forecast wear progression based on usage
- **Anomaly Detection**: Spot weird/unexpected wear patterns

## Input Data (What We'll Use)

- Tool type and ID
- Machining process (milling, drilling, etc.)
- Sensor readings like vibration, force, AE (acoustic emission), and temperature
- Machine parameters (spindle speed, feed rate, etc.)
- Historical wear measurements (flank wear, cracks, etc.)

## Expected Outputs

- Wear level (numerical or categorical)
- Tool condition label (Healthy / Needs Replacement / Critical)
- Estimated remaining tool life (optional)

---

This project is part of a bigger push toward using data and machine learning to improve industrial processes — especially in precision manufacturing like engine building.
## Running Tests

Install pytest and run:

```bash
pytest
```

