# Data Science Portfolio

This repository contains learning notebooks, Python scripts, and a SpaceX Falcon 9 launch analysis project from the IBM Data Science specialization.

## Repository Structure

```text
notebooks/
  intro/      Introductory data science and stock/revenue extraction assignments.
  spacex/     SpaceX data collection, wrangling, EDA, mapping, and modeling notebooks.
apps/
  spacex_dash/  Dash dashboard for SpaceX launch outcomes.
src/
  testrepo/  Reusable Python helpers.
data/
  processed/ Processed CSV files used by notebooks and the Dash app.
scripts/     Small beginner Python practice scripts.
```

## SpaceX Project Flow

The SpaceX notebooks are organized as an end-to-end project:

1. Data collection from the SpaceX API
2. Web scraping launch records
3. Data wrangling
4. Exploratory analysis with SQL
5. Exploratory data visualization
6. Launch site mapping with Folium
7. Machine learning prediction
8. Interactive dashboard with Dash

## Data

Processed SpaceX datasets live in `data/processed/`:

- `spacex_launch_dash.csv`
- `spacex_launch_geo.csv`

## Running the Dash App

Install the required packages, then run:

```bash
python apps/spacex_dash/app.py
```

The dashboard reads `data/processed/spacex_launch_dash.csv`.
