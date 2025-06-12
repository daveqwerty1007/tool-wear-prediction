# Tool Wear Prediction in Engine Manufacturing

## What are we trying to do?

We want to know how worn each cutting tool is so we can swap it out before it ruins a part. We'll crunch the sensor data coming off the machines and build models that tell us if a tool is still healthy or on its last legs.

## Why should anyone care?
When a tool fails unexpectedly, it wastes time and money. If we can spot the wear early, the line keeps humming and we avoid a lot of headaches.

## Approaches we can try
- **Classification** – tag a tool as Healthy, Worn, or Broken
- **Regression** – estimate the remaining life in hours or the amount of wear
- **Time Series** – track how wear progresses job after job
- **Anomaly Detection** – catch weird patterns we didn't expect

## What data do we have to work with?
- Tool ID and type
- Which machining process was used
- Vibration, force, acoustic emission, and temperature readings
- Machine settings like spindle speed and feed rate
- Measurements of wear such as flank wear or cracks

## What do we want out?
- A wear level for each tool
- A friendly label like "Needs Replacement"
- Maybe an estimate of remaining life

---

This repo is one step toward smarter, predictive maintenance in manufacturing.

## Notebooks
The `notebooks/` folder contains a simple workflow:
- `01_EDA.ipynb` – peek at the data, plot histograms, and check for outliers
- `02_Features.ipynb` – scale values, create a few helper features, and save a cleaned CSV
- `03_Modeling.ipynb` – try a random‑forest classifier and a Ridge regressor

## Data
Raw CSV files live under `data/raw/`. They aren't included here. Grab the original
Vicomtech Tool Wear dataset from their repository or fork and drop the CSV in that folder.

## Running the tests
Install the dependencies and run `pytest`:
```bash
pip install -r requirements.txt
pytest -q
```
