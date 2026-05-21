# Village Economic Growth Intelligence System
### Kritter Software Technologies — Candidate Assignment

> Identifying India's Top 100 Fastest-Growing Villages (2020–2025) using Satellite Imagery, Geospatial AI, and Public Data

---

## Overview

This system builds a production-grade geospatial intelligence pipeline to rank India's villages by economic growth using observable satellite signals — nighttime lights, built-up expansion, NDVI trends, road density, and land-use change.

## Project Structure

```
village_economic_growth/
├── README.md
├── requirements.txt
├── src/
│   ├── ingestion/         # Data collection from GEE, OSM, Census
│   ├── preprocessing/     # Cloud masking, compositing, normalization
│   ├── features/          # Feature engineering per village
│   ├── modeling/          # Economic Growth Score computation
│   ├── visualization/     # Maps, charts, dashboards
│   └── export/            # CSV, GeoJSON, Shapefile outputs
├── notebooks/
│   └── village_growth_pipeline.ipynb   # Full Colab notebook
├── data/
│   ├── raw/               # Source data (GEE exports, OSM)
│   ├── processed/         # Cleaned & feature-engineered data
│   └── outputs/           # Final ranked dataset + maps
└── reports/
    └── presentation.html  # 5-slide summary deck
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python src/pipeline.py

# 3. Open interactive dashboard
open data/outputs/village_growth_map.html
```

## Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| VIIRS Nighttime Lights | NOAA/GEE (`NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`) | Electrification, commerce |
| Sentinel-2 MSI | ESA/GEE (`COPERNICUS/S2_SR_HARMONIZED`) | NDVI, built-up expansion |
| Dynamic World | Google/GEE (`GOOGLE/DYNAMICWORLD/V1`) | Land use/land cover |
| OpenStreetMap | Overpass API | Road density, infrastructure |
| India Census 2011/2021 | data.gov.in | Population, admin metadata |
| Village Boundaries | Datameet / GADM | Spatial aggregation |

## Economic Growth Score Formula

```
EGS = 0.35 × ΔNightlight + 0.25 × ΔBuilt-up + 0.20 × ΔRoad_Density + 0.10 × ΔNDVI + 0.10 × ΔLULC
```

All components normalized to [0,1] using min-max scaling before weighting.

## Key Outputs

- `data/outputs/top100_villages.csv` — Ranked dataset with scores
- `data/outputs/top100_villages.geojson` — Spatial output
- `data/outputs/village_growth_map.html` — Interactive Folium map
- `reports/presentation.html` — Executive summary slides
