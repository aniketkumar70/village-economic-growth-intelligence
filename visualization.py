import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap, Fullscreen
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import logging

log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent
DATA_OUT = ROOT / 'data' / 'outputs'
DATA_OUT.mkdir(parents=True, exist_ok=True)


# ── COLOR PALETTES ───────────────────────────────────────────
TIER_COLORS = {
    'Elite':     '#FF3A00',
    'Very High': '#FF8C00',
    'High':      '#FFD700',
    'Moderate':  '#4CAF50',
    'Baseline':  '#90CAF9',
}

EGS_COLORSCALE = [
    [0.0,  '#0D1B2A'],
    [0.2,  '#1B3A5C'],
    [0.4,  '#1565C0'],
    [0.6,  '#00ACC1'],
    [0.8,  '#FFD600'],
    [1.0,  '#FF3A00'],
]


# ══════════════════════════════════════════════════════════════
# 1. INTERACTIVE FOLIUM MAP
# ══════════════════════════════════════════════════════════════

def create_interactive_map(top100: pd.DataFrame, all_villages: pd.DataFrame) -> str:
    """
    Create a rich interactive Folium map with:
    - Top 100 village markers (color-coded by EGS)
    - Heatmap layer of all villages
    - Cluster layer for dense regions
    - Popup cards with full village data
    - Layer toggle controls
    """
    log.info('Creating interactive map...')

    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles=None,
    )

    # ── Base tile layers ─────────────────────────────────────
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB',
        name='Dark (Default)',
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles='CartoDB positron',
        name='Light',
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        control=True,
    ).add_to(m)

    # ── Heatmap layer — all villages ─────────────────────────
    heat_data = all_villages[['latitude', 'longitude', 'economic_growth_score']].values.tolist()
    heatmap_layer = folium.FeatureGroup(name='Growth Heatmap (All Villages)', show=True)
    HeatMap(
        heat_data,
        radius=12,
        blur=18,
        max_zoom=10,
        gradient={
            '0.0': '#001529',
            '0.3': '#003c8f',
            '0.5': '#0277bd',
            '0.7': '#f9a825',
            '1.0': '#ff3a00',
        },
    ).add_to(heatmap_layer)
    heatmap_layer.add_to(m)

    # ── Top 100 markers ──────────────────────────────────────
    top100_layer = folium.FeatureGroup(name='🏆 Top 100 Villages', show=True)

    for _, row in top100.iterrows():
        tier  = str(row.get('growth_tier', 'High'))
        color = TIER_COLORS.get(tier, '#FF8C00')
        rank  = int(row['national_rank'])
        score = float(row['economic_growth_score'])

        # Marker size by rank (top villages = bigger)
        radius = max(6, 18 - rank * 0.12)

        popup_html = f"""
        <div style="font-family: 'Segoe UI', sans-serif; min-width: 260px; padding: 4px;">
          <div style="background: linear-gradient(135deg, #0D1B2A, #1565C0);
                      color: white; padding: 10px 14px; border-radius: 8px 8px 0 0;
                      margin: -4px -4px 0 -4px;">
            <div style="font-size: 18px; font-weight: 700;">#{rank} {row['village_name']}</div>
            <div style="font-size: 12px; opacity: 0.85;">{row['district']}, {row['state']}</div>
          </div>
          <div style="padding: 10px 14px; background: #1a1a2e; color: #e0e0e0;
                      border-radius: 0 0 8px 8px;">
            <div style="font-size: 22px; font-weight: 800; color: {color};">
              {score:.1f} <span style="font-size: 12px; color: #aaa;">/ 100 EGS</span>
            </div>
            <hr style="border-color: #333; margin: 8px 0;">
            <table style="font-size: 12px; width: 100%; border-collapse: collapse;">
              <tr><td style="color:#aaa; padding: 2px 0;">💡 Nightlight Growth</td>
                  <td style="text-align:right; color:#FFD600; font-weight:600;">
                    {row['ntl_growth_pct']:+.1f}%</td></tr>
              <tr><td style="color:#aaa; padding: 2px 0;">🏗️ Built-up Expansion</td>
                  <td style="text-align:right; color:#FF8C00; font-weight:600;">
                    {row['built_growth_abs']*100:+.1f}%</td></tr>
              <tr><td style="color:#aaa; padding: 2px 0;">🛣️ Road Growth</td>
                  <td style="text-align:right; color:#4CAF50; font-weight:600;">
                    {row['road_growth_pct']:+.1f}%</td></tr>
              <tr><td style="color:#aaa; padding: 2px 0;">🌾 NDVI Change</td>
                  <td style="text-align:right; color:#81C784; font-weight:600;">
                    {row['ndvi_change']:+.3f}</td></tr>
            </table>
            <div style="margin-top: 8px; padding: 4px 8px; background: {color}22;
                        border-left: 3px solid {color}; border-radius: 4px;
                        font-size: 11px; color: {color};">
              {tier} Growth Tier | Confidence: {row['confidence_score']:.0f}%
            </div>
          </div>
        </div>
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"#{rank} {row['village_name']} — EGS: {score:.1f}",
        ).add_to(top100_layer)

    top100_layer.add_to(m)

    # ── Cluster layer — Top 100 ──────────────────────────────
    cluster_layer = folium.FeatureGroup(name='📍 Village Clusters', show=False)
    mc = MarkerCluster()
    for _, row in top100.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            tooltip=f"#{int(row['national_rank'])} {row['village_name']}",
        ).add_to(mc)
    mc.add_to(cluster_layer)
    cluster_layer.add_to(m)

    # ── Legend ───────────────────────────────────────────────
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: rgba(13,27,42,0.95); color: white;
                padding: 16px 20px; border-radius: 12px;
                border: 1px solid #1565C0; font-family: 'Segoe UI', sans-serif;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
      <div style="font-size: 14px; font-weight: 700; margin-bottom: 10px;
                  letter-spacing: 1px; color: #90CAF9;">GROWTH TIER</div>
      """ + ''.join([
        f'<div style="display:flex; align-items:center; margin: 5px 0;">'
        f'<div style="width:12px; height:12px; border-radius:50%; '
        f'background:{c}; margin-right:10px;"></div>'
        f'<span style="font-size:12px;">{t}</span></div>'
        for t, c in TIER_COLORS.items()
    ]) + """
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Controls ─────────────────────────────────────────────
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    Fullscreen(position='topleft').add_to(m)

    # ── Title ────────────────────────────────────────────────
    title_html = """
    <div style="position:fixed; top:12px; left:50%; transform:translateX(-50%);
                z-index:9999; background:rgba(13,27,42,0.9);
                color:white; padding:10px 24px; border-radius:8px;
                font-family:'Segoe UI',sans-serif; text-align:center;
                border: 1px solid rgba(21,101,192,0.5);">
      <div style="font-size:16px; font-weight:700; letter-spacing:1px;">
        🇮🇳 Village Economic Growth Intelligence — Top 100 (2020–2025)
      </div>
      <div style="font-size:11px; color:#90CAF9; margin-top:2px;">
        Economic Growth Score • Satellite + GIS Analysis
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Save
    map_path = DATA_OUT / 'village_growth_map.html'
    m.save(str(map_path))
    log.info(f'Interactive map saved: {map_path}')
    return str(map_path)


# ══════════════════════════════════════════════════════════════
# 2. PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════

def create_dashboard_charts(top100: pd.DataFrame, all_villages: pd.DataFrame) -> str:
    """
    Create a multi-panel Plotly dashboard with:
    - Top 20 bar chart
    - State-wise distribution
    - Score components radar
    - NTL vs Built-up scatter
    - Growth tier donut
    - Score distribution
    """
    log.info('Creating dashboard charts...')

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Top 20 Villages by Economic Growth Score',
            'State-wise Share of Top 100 Villages',
            'Growth Signal Components — Top 20 Average',
            'Nightlight Growth vs Built-up Expansion',
            'Growth Tier Distribution',
            'EGS Distribution — All Villages',
        ),
        specs=[
            [{'type': 'bar'},            {'type': 'bar'}],
            [{'type': 'bar'},            {'type': 'scatter'}],
            [{'type': 'pie'},            {'type': 'histogram'}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ── Panel 1: Top 20 Bar ──────────────────────────────────
    top20 = top100.head(20)
    colors = [TIER_COLORS.get(str(t), '#FF8C00') for t in top20['growth_tier']]
    fig.add_trace(go.Bar(
        x=top20['economic_growth_score'],
        y=top20.apply(lambda r: f"#{int(r['national_rank'])} {r['village_name']}", axis=1),
        orientation='h',
        marker_color=colors,
        text=top20['state'],
        textposition='inside',
        textfont=dict(size=10, color='white'),
        hovertemplate='<b>%{y}</b><br>EGS: %{x:.1f}<br>State: %{text}<extra></extra>',
    ), row=1, col=1)

    # ── Panel 2: State Distribution ─────────────────────────
    state_counts = top100['state'].value_counts().head(12)
    fig.add_trace(go.Bar(
        x=state_counts.values,
        y=state_counts.index,
        orientation='h',
        marker_color='#1565C0',
        marker_line_color='#42A5F5',
        marker_line_width=1,
        hovertemplate='<b>%{y}</b>: %{x} villages<extra></extra>',
    ), row=1, col=2)

    # ── Panel 3: Component breakdown ────────────────────────
    comp_means = top20[[
        'ntl_growth_norm', 'builtup_growth_norm',
        'road_density_norm', 'ndvi_productivity_norm', 'lulc_change_norm'
    ]].mean()
    comp_labels = ['Nightlights', 'Built-up', 'Roads', 'Agriculture', 'Land Use']
    comp_colors = ['#FFD600', '#FF6D00', '#2196F3', '#4CAF50', '#AB47BC']
    fig.add_trace(go.Bar(
        x=comp_means.values,
        y=comp_labels,
        orientation='h',
        marker_color=comp_colors,
        hovertemplate='<b>%{y}</b>: %{x:.1f}/100<extra></extra>',
    ), row=2, col=1)

    # ── Panel 4: NTL vs Built-up scatter ─────────────────────
    sample = all_villages.sample(min(2000, len(all_villages)), random_state=42)
    fig.add_trace(go.Scatter(
        x=sample['ntl_growth_pct'],
        y=sample['built_growth_abs'] * 100,
        mode='markers',
        marker=dict(
            size=4,
            color=sample['economic_growth_score'],
            colorscale='Plasma',
            opacity=0.6,
            colorbar=dict(title='EGS', x=1.02),
        ),
        hovertemplate='NTL: %{x:.1f}%<br>Built-up: %{y:.1f}%<extra></extra>',
    ), row=2, col=2)

    # Add Top 100 highlight
    fig.add_trace(go.Scatter(
        x=top100['ntl_growth_pct'],
        y=top100['built_growth_abs'] * 100,
        mode='markers',
        marker=dict(size=8, color='#FF3A00', symbol='star', line=dict(width=1, color='white')),
        name='Top 100',
        hovertemplate='<b>%{text}</b><br>NTL: %{x:.1f}%<br>Built-up: %{y:.1f}%<extra></extra>',
        text=top100['village_name'],
    ), row=2, col=2)

    # ── Panel 5: Tier donut ──────────────────────────────────
    tier_counts = all_villages['growth_tier'].value_counts()
    fig.add_trace(go.Pie(
        labels=tier_counts.index,
        values=tier_counts.values,
        hole=0.5,
        marker_colors=[TIER_COLORS.get(t, '#90CAF9') for t in tier_counts.index],
        hovertemplate='<b>%{label}</b>: %{value:,} villages (%{percent})<extra></extra>',
    ), row=3, col=1)

    # ── Panel 6: Score distribution ─────────────────────────
    threshold = top100['economic_growth_score'].min()
    fig.add_trace(go.Histogram(
        x=all_villages['economic_growth_score'],
        nbinsx=50,
        marker_color='#1565C0',
        marker_line_color='#42A5F5',
        marker_line_width=0.5,
        hovertemplate='Score: %{x:.1f}<br>Count: %{y}<extra></extra>',
    ), row=3, col=2)
    # Mark Top 100 threshold with a shape on the correct axis
    fig.add_shape(
        type='line',
        x0=threshold, x1=threshold,
        y0=0, y1=1,
        yref='paper',
        xref='x6',
        line=dict(color='#FF3A00', dash='dash', width=2),
    )
    fig.add_annotation(
        x=threshold, y=0.05,
        xref='x6', yref='paper',
        text='Top 100 Threshold',
        showarrow=True, arrowhead=2,
        font=dict(color='#FF3A00', size=10),
        arrowcolor='#FF3A00',
    )

    # ── Layout ───────────────────────────────────────────────
    fig.update_layout(
        height=1200,
        template='plotly_dark',
        paper_bgcolor='#0D1B2A',
        plot_bgcolor='#0D1B2A',
        font=dict(family='Segoe UI, sans-serif', color='#E0E0E0'),
        title=dict(
            text='Village Economic Growth Intelligence Dashboard — India 2020→2025',
            x=0.5, font=dict(size=18, color='white'),
        ),
        showlegend=False,
    )

    chart_path = DATA_OUT / 'growth_dashboard.html'
    fig.write_html(str(chart_path))
    log.info(f'Dashboard saved: {chart_path}')
    return str(chart_path)


# ══════════════════════════════════════════════════════════════
# 3. MATPLOTLIB STATIC CHARTS
# ══════════════════════════════════════════════════════════════

def create_static_charts(top100: pd.DataFrame) -> str:
    """Generate publication-quality static charts for the report."""
    log.info('Creating static charts...')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0D1B2A')
    plt.rcParams.update({
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
    })

    for ax in axes.flat:
        ax.set_facecolor('#1B2A3B')
        for spine in ax.spines.values():
            spine.set_color('#334455')

    # ── (a) Top 10 bar chart ─────────────────────────────────
    ax = axes[0, 0]
    top10 = top100.head(10)
    labels = [f"#{int(r['national_rank'])} {r['village_name']}" for _, r in top10.iterrows()]
    scores = top10['economic_growth_score'].values
    bar_colors = [TIER_COLORS.get(str(t), '#FF8C00') for t in top10['growth_tier']]
    bars = ax.barh(labels, scores, color=bar_colors, edgecolor='#334455', linewidth=0.5)
    ax.set_xlabel('Economic Growth Score (0–100)', color='#90CAF9')
    ax.set_title('Top 10 Fastest-Growing Villages', color='white', fontweight='bold', pad=12)
    ax.invert_yaxis()
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=9, color='#FFD600')

    # ── (b) State distribution ───────────────────────────────
    ax = axes[0, 1]
    state_c = top100['state'].value_counts().head(10)
    cmap = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(state_c)))
    ax.barh(state_c.index, state_c.values, color=cmap)
    ax.set_xlabel('Number of Villages in Top 100', color='#90CAF9')
    ax.set_title('State-wise Top 100 Distribution', color='white', fontweight='bold', pad=12)
    ax.invert_yaxis()

    # ── (c) Score components stacked bar ────────────────────
    ax = axes[1, 0]
    comp_cols = ['ntl_growth_norm', 'builtup_growth_norm',
                 'road_density_norm', 'ndvi_productivity_norm', 'lulc_change_norm']
    comp_labels = ['Nightlights (35%)', 'Built-up (25%)', 'Roads (20%)',
                   'Agriculture (10%)', 'Land Use (10%)']
    comp_colors_mpl = ['#FFD600', '#FF6D00', '#2196F3', '#4CAF50', '#AB47BC']
    comp_means = top100[comp_cols].mean().values

    bars2 = ax.bar(comp_labels, comp_means, color=comp_colors_mpl,
                   edgecolor='#334455', linewidth=0.5)
    ax.set_ylabel('Average Score (0–100)', color='#90CAF9')
    ax.set_title('EGS Component Scores — Top 100 Average', color='white',
                 fontweight='bold', pad=12)
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars2, comp_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=9, color='white')

    # ── (d) Growth scatter — NTL vs built-up ────────────────
    ax = axes[1, 1]
    sc = ax.scatter(
        top100['ntl_growth_pct'], top100['built_growth_abs'] * 100,
        c=top100['economic_growth_score'],
        cmap='plasma', s=top100['economic_growth_score'] * 2 + 20,
        alpha=0.8, edgecolors='white', linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax, label='Economic Growth Score').ax.yaxis.label.set_color('white')
    ax.set_xlabel('Nightlight Growth (%)', color='#90CAF9')
    ax.set_ylabel('Built-up Area Growth (%)', color='#90CAF9')
    ax.set_title('Nightlight vs Built-up Growth — Top 100', color='white',
                 fontweight='bold', pad=12)

    plt.suptitle('Village Economic Growth Intelligence System — India 2020→2025',
                 fontsize=14, color='white', fontweight='bold', y=1.01)
    plt.tight_layout()

    chart_path = DATA_OUT / 'growth_charts.png'
    plt.savefig(str(chart_path), dpi=150, bbox_inches='tight',
                facecolor='#0D1B2A')
    plt.close()
    log.info(f'Static charts saved: {chart_path}')
    return str(chart_path)


def run_visualizations(top100: pd.DataFrame, all_villages: pd.DataFrame) -> dict:
    """Run all visualization modules."""
    paths = {}
    paths['map']      = create_interactive_map(top100, all_villages)
    paths['dashboard'] = create_dashboard_charts(top100, all_villages)
    paths['charts']   = create_static_charts(top100)
    return paths
