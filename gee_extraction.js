/**
 * ============================================================
 * Village Economic Growth Intelligence System
 * Google Earth Engine Script — Data Extraction
 * ============================================================
 * Purpose: Extract village-level satellite indicators for 2020 & 2025
 * Platform: Google Earth Engine Code Editor (code.earthengine.google.com)
 * Author: Kritter Assignment Submission
 * ============================================================
 */

// ── CONFIG ──────────────────────────────────────────────────
var START_2020 = '2020-01-01';
var END_2020   = '2020-12-31';
var START_2025 = '2024-06-01';
var END_2025   = '2025-05-31';
var INDIA_BOUNDS = ee.Geometry.Rectangle([68.1, 8.0, 97.4, 37.6]);

// ── 1. VILLAGE BOUNDARIES ───────────────────────────────────
// Using GADM level-3 (sub-district) as village proxy
// Replace with actual village shapefile if available
var villages = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME', 'India'))
  .limit(5000); // Process in batches for full India run

print('Village count (sample):', villages.size());

// ── 2. VIIRS NIGHTTIME LIGHTS ────────────────────────────────
/**
 * Band: avg_rad — Average DNB radiance (nanoWatts/cm²/sr)
 * Higher values → more economic activity, electrification
 */
function getNightlights(startDate, endDate) {
  return ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
    .filterDate(startDate, endDate)
    .filterBounds(INDIA_BOUNDS)
    .select('avg_rad')
    .median() // Median composite to reduce noise/cloud artifacts
    .rename('nightlight');
}

var ntl_2020 = getNightlights(START_2020, END_2020);
var ntl_2025 = getNightlights(START_2025, END_2025);
var ntl_delta = ntl_2025.subtract(ntl_2020).rename('ntl_delta');

// ── 3. SENTINEL-2 — NDVI & BUILT-UP INDEX ────────────────────
/**
 * NDVI = (NIR - RED) / (NIR + RED)  →  Vegetation health & cropping intensity
 * NDBI = (SWIR - NIR) / (SWIR + NIR) →  Built-up proxy (positive = urban)
 * MNDWI = (GREEN - SWIR) / (GREEN + SWIR) → Water mask
 */
function cloudMaskS2(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000); // Scale reflectance
}

function getSentinel2(startDate, endDate) {
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filterBounds(INDIA_BOUNDS)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(cloudMaskS2)
    .median();

  var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi');
  var ndbi = s2.normalizedDifference(['B11', 'B8']).rename('ndbi');
  var mndwi = s2.normalizedDifference(['B3', 'B11']).rename('mndwi');
  var bsi = s2.expression(
    '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
    { SWIR: s2.select('B11'), RED: s2.select('B4'),
      NIR: s2.select('B8'), BLUE: s2.select('B2') }
  ).rename('bsi'); // Bare Soil Index — construction proxy

  return ee.Image([ndvi, ndbi, mndwi, bsi]);
}

var s2_2020 = getSentinel2(START_2020, END_2020);
var s2_2025 = getSentinel2(START_2025, END_2025);

// ── 4. DYNAMIC WORLD — LAND USE / LAND COVER ─────────────────
/**
 * Google Dynamic World: 9-class probabilistic LULC at 10m
 * Classes: water, trees, grass, flooded, crops, shrub,
 *          built, bare, snow_ice
 * We extract built-up probability as urban expansion signal
 */
function getDynamicWorld(startDate, endDate) {
  return ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterDate(startDate, endDate)
    .filterBounds(INDIA_BOUNDS)
    .select(['built', 'crops', 'trees'])
    .mean(); // Mean probability over period
}

var dw_2020 = getDynamicWorld(START_2020, END_2020);
var dw_2025 = getDynamicWorld(START_2025, END_2025);
var dw_delta_built = dw_2025.select('built')
  .subtract(dw_2020.select('built'))
  .rename('built_delta');

// ── 5. AGGREGATE FEATURES PER VILLAGE ───────────────────────
var combinedImage = ee.Image([
  ntl_2020.rename('ntl_2020'),
  ntl_2025.rename('ntl_2025'),
  ntl_delta,
  s2_2020.select('ndvi').rename('ndvi_2020'),
  s2_2025.select('ndvi').rename('ndvi_2025'),
  s2_2020.select('ndbi').rename('ndbi_2020'),
  s2_2025.select('ndbi').rename('ndbi_2025'),
  s2_2020.select('bsi').rename('bsi_2020'),
  s2_2025.select('bsi').rename('bsi_2025'),
  dw_2020.select('built').rename('built_prob_2020'),
  dw_2025.select('built').rename('built_prob_2025'),
  dw_delta_built,
  dw_2020.select('crops').rename('crops_2020'),
  dw_2025.select('crops').rename('crops_2025'),
]);

// Reduce to village-level means using zonal statistics
var villageStats = combinedImage.reduceRegions({
  collection: villages,
  reducer: ee.Reducer.mean().combine({
    reducer2: ee.Reducer.stdDev(),
    sharedInputs: true
  }),
  scale: 500, // 500m resolution for efficient processing
  crs: 'EPSG:4326',
  tileScale: 4  // Parallelism boost for large collections
});

// ── 6. COMPUTE GROWTH INDICATORS IN GEE ─────────────────────
var villageIndicators = villageStats.map(function(f) {
  var ntl20 = ee.Number(f.get('ntl_2020_mean')).max(0.01); // Avoid div-by-zero
  var ntl25 = ee.Number(f.get('ntl_2025_mean')).max(0.01);
  var ntlGrowth = ntl25.subtract(ntl20).divide(ntl20).multiply(100); // % change

  var ndvi20 = ee.Number(f.get('ndvi_2020_mean'));
  var ndvi25 = ee.Number(f.get('ndvi_2025_mean'));
  var ndviChange = ndvi25.subtract(ndvi20);

  var ndbi20 = ee.Number(f.get('ndbi_2020_mean'));
  var ndbi25 = ee.Number(f.get('ndbi_2025_mean'));
  var ndbiGrowth = ndbi25.subtract(ndbi20); // Positive = more built-up

  var built20 = ee.Number(f.get('built_prob_2020_mean'));
  var built25 = ee.Number(f.get('built_prob_2025_mean'));
  var builtGrowth = built25.subtract(built20).multiply(100); // % prob increase

  return f.set({
    'ntl_growth_pct': ntlGrowth,
    'ndvi_change': ndviChange,
    'ndbi_growth': ndbiGrowth,
    'built_area_growth_pct': builtGrowth,
    'ntl_2020': ntl20,
    'ntl_2025': ntl25,
  });
});

// ── 7. EXPORT ────────────────────────────────────────────────
Export.table.toDrive({
  collection: villageIndicators,
  description: 'village_economic_indicators_2020_2025',
  folder: 'VillageGrowth',
  fileNamePrefix: 'village_indicators',
  fileFormat: 'CSV',
  selectors: [
    'ADM2_NAME', 'ADM1_NAME',           // District, State
    'ntl_2020', 'ntl_2025',             // Nightlights raw
    'ntl_growth_pct',                    // Nightlight % growth
    'ndvi_2020_mean', 'ndvi_2025_mean',  // NDVI
    'ndvi_change',                        // NDVI change
    'ndbi_2020_mean', 'ndbi_2025_mean',  // NDBI
    'ndbi_growth',                        // Built-up growth
    'bsi_2020_mean', 'bsi_2025_mean',    // Bare soil index
    'built_prob_2020_mean', 'built_prob_2025_mean',
    'built_area_growth_pct',             // Dynamic world built growth
    'crops_2020_mean', 'crops_2025_mean' // Crop area probability
  ]
});

// ── 8. QUICK MAP VISUALIZATION ───────────────────────────────
Map.centerObject(INDIA_BOUNDS, 5);

Map.addLayer(ntl_delta, {
  min: -10, max: 50,
  palette: ['#0d0221', '#1b1b8a', '#3366cc', '#00cc44', '#ffff00', '#ff6600']
}, 'Nightlight Delta 2020→2025');

Map.addLayer(dw_delta_built, {
  min: -0.1, max: 0.3,
  palette: ['#ffffff', '#ffffcc', '#ffcc00', '#ff6600', '#cc0000']
}, 'Built-up Area Growth');

Map.addLayer(s2_2025.select(['B4', 'B3', 'B2']), {
  min: 0, max: 0.3,
  gamma: 1.4
}, 'Sentinel-2 True Color 2025');

print('Pipeline complete. Check Tasks tab to monitor export.');
