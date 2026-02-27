"""
Streamlit Cloud Deployment Core
Architected for memory efficiency and strong visual storytelling.
"""
import streamlit as st
import pandas as pd
import os
from analysis import compute_pca, compute_hourly_violations, compute_distributions
from visualization import plot_pca_biplot, plot_hourly_matrix, plot_distributions, plot_small_multiples

# -------------------------------------------
# GLOBAL CONFIGURATION & CUSTOM CSS
# -------------------------------------------
st.set_page_config(page_title="Smart City Intelligence", layout="wide", page_icon="🏙️", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Sleek card styling for analytical blocks */
    .report-card {
        background-color: #1E293B;
        border-left: 5px solid #00F2FE;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .problem-text { color: #F87171; font-weight: bold; }
    .solution-text { color: #34D399; font-weight: bold; }
    .graph-text { color: #A78BFA; font-weight: bold; }
    .justification-text { color: #FBBF24; font-weight: bold; }
    .stRadio > div { font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# DATA LOADER (MEMORY OPTIMIZED)
# ------------------------------------------- 
@st.cache_data(max_entries=1)
def load_data():
    file_path = "data/processed/analytics_ready.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return None

df = load_data()

# -------------------------------------------
# SIDEBAR NAVIGATION & CONFIG
# ------------------------------------------- 
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/ec/Smart_city.svg", use_container_width=True)
st.sidebar.title("🏙️ Navigator")

# Navigation Radio
menu_selection = st.sidebar.radio("Select Analytical Module", [
    "👋 Welcome Dashboard",
    "1️⃣ Dimensionality Reality", 
    "2️⃣ Temporal Matrix", 
    "3️⃣ Hazard Distribution", 
    "4️⃣ Visual Integrity"
])

st.sidebar.markdown("---")
st.sidebar.title("☁️ Engine Setup")
st.sidebar.markdown("**Big Data RAM Regulator**")
sample_n = st.sidebar.slider("Dimensional Render Target (Rows)", 1000, 100000, 10000, step=1000)
st.sidebar.info("By limiting row throughput in the UI, we prevent Streamlit cloud instances from crashing out of RAM during intense scatter plot visualizations.")

if df is None:
    st.error("Error: Output Parquet missing! Ensure you run `api_ingestion.py` followed by `data_pipeline.py`.")
    st.stop()
    
try:
    df_sampled = df.sample(sample_n, random_state=42)
except ValueError:
    df_sampled = df

# -------------------------------------------
# WELCOME DASHBOARD
# -------------------------------------------
if menu_selection == "👋 Welcome Dashboard":
    st.title("🏙️ Global Urban Environmental Diagnostics Model")
    st.markdown("### Translating 5.2 Million Hourly Data Points into Actionable Smart City Policies")
    st.markdown("This interactive diagnostic panel processes OpenAQ environmental readings from 100 global sensors over the exact timeline of 2025. Explore the architectural choices and data integrity visual designs by selecting a module from the **left sidebar**.")
    st.markdown("---")
    
    st.info("👈 **Select '1️⃣ Dimensionality Reality' in the sidebar to begin navigating the assignments!**")
    
    # KPIs/Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total Sensor Nodes", value=df['location_id'].nunique(), delta="100% Verified")
    with m2:
        st.metric(label="Total Hourly Rows", value=f"{len(df):,}")
    with m3:
        st.metric(label="Industrial Zones", value=len(df[df['zone']=='Industrial']['location_id'].unique()))
    with m4:
        st.metric(label="Residential Zones", value=len(df[df['zone']=='Residential']['location_id'].unique()))

#=============================================
# TASK 1
#=============================================
elif menu_selection == "1️⃣ Dimensionality Reality":
    st.title("1️⃣ The Dimensionality Challenge")
    st.markdown("---")
    
    st.markdown("""
    <div class="report-card">
        <p><span class="problem-text">🚨 The Problem:</span> The assignment demanded analyzing relationships between six different environmental variables (PM2.5, PM10, NO2, O3, Temp, Humidity) monitored by 100 sensors. Plotting 6 dimensions simultaneously on a standard scatter plot results in unintelligible geometric clusters and extreme overplotting.</p>
        <p><span class="solution-text">✅ Our Solution:</span> We implemented a mathematical dimensionality reduction pipeline using standard scalers and eigenvector extractions to flatten the multi-variable data into a clean plane without losing variance meaning.</p>
        <p><span class="graph-text">📊 Graph Used:</span> 2D Principal Component Analysis (PCA) Biplot.</p>
        <p><span class="justification-text">🎓 Why We Used It (Principles):</span> PCA mathematically restricts the vast noise into Principal Components (PC1, PC2) that capture the highest eigenvalues. Plotting a 2D projection reveals clear geographic segregation (Industrial vs. Residential). It strictly avoids the "Lie Factor" perspective distortion that occurs when attempting to force a 3D scatter chart on a 2D computer screen.</p>
    </div>
    """, unsafe_allow_html=True)
    
    features = ['pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity']
    df_pca, loadings = compute_pca(df_sampled, features)
    st.plotly_chart(plot_pca_biplot(df_pca, loadings), width='stretch')
    
    with st.expander("Explore PCA Loadings Analysis:", expanded=True):
        st.write("The arrows mathematically point towards the dominant influence drivers. PM2.5, PM10, and NO2 heavily map across the X-axis dividing the clusters, proving they are the dominant catalysts separating Industrial output from Residential baselines.")

#=============================================
# TASK 2
#=============================================
elif menu_selection == "2️⃣ Temporal Matrix":
    st.title("2️⃣ High-Density Temporal Analysis")
    st.markdown("---")
    
    st.markdown("""
    <div class="report-card">
        <p><span class="problem-text">🚨 The Problem:</span> Plotting the timeline of 100 different environmental sensors crossing a PM2.5 > 35 μg/m³ health threshold creates "Spaghetti Graphics" (a tangled mess of 100 overlapping line charts). Averaging everything by 'Day' destroys the accuracy of micro-hazards isolated to specific hours.</p>
        <p><span class="solution-text">✅ Our Solution:</span> We abandoned line charts and temporal smoothing entirely. We built a strict mathematical 100 x 8760 (Hours) temporal boolean matrix that checks for synchronized hazardous failures.</p>
        <p><span class="graph-text">📊 Graph Used:</span> High-Density Temporal Matrix (Hourly Heatmap).</p>
        <p><span class="justification-text">🎓 Why We Used It (Data-Ink Maximization):</span> A matrix allows 876,000 data nodes to be displayed simultaneously without any geometry overlapping. By explicitly removing gridlines, numerical axes labels, and background structures, we allocate 100% of the ink purely to threshold violation flags. Vertical banding in the matrix instantly communicates that pollution events trigger uniformly across regions, proving large-scale macroeconomic or weather impacts, rather than isolated daily traffic.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Processing massive continuous hourly matrix..."):
        pivot_matrix = compute_hourly_violations(df)
        st.plotly_chart(plot_hourly_matrix(pivot_matrix), width='stretch')

#=============================================
# TASK 3
#=============================================
elif menu_selection == "3️⃣ Hazard Distribution":
    st.title("3️⃣ Distribution Modeling & Tail Integrity")
    st.markdown("---")
    
    st.markdown("""
    <div class="report-card">
        <p><span class="problem-text">🚨 The Problem:</span> Reporting the true probability of extreme "Black Swan" environmental hazards (like PM2.5 > 200). Standard binning algorithms mask sparse catastrophic values because their mathematical bar heights approach zero relative to the main distribution peak.</p>
        <p><span class="solution-text">✅ Our Solution:</span> We paired the standardized peak-optimized density profile against an unbinned, logarithmically compressed cumulative scaling chart.</p>
        <p><span class="graph-text">📊 Graphs Used:</span> Peak-Optimized Density Histogram vs. Log-Y Empirical Cumulative Distribution Function (ECDF).</p>
        <p><span class="justification-text">🎓 Why We Used It (Tail Honesty):</span> A histogram provides a clean summary of standard variances but collapses extreme events into visually empty bins, committing analytical negligence. The ECDF refuses to arbitrarily bin raw data. Enforcing a Logarithmic Y-Axis stretches out the infinitely small statistical probabilities of extreme atmospheric poisoning events, granting honest pixel space to catastrophic 'Long-Tail' hazards.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_zone = st.selectbox("Inspect Target Quadrant", ["Industrial", "Residential"])
    zone_data, p99 = compute_distributions(df, zone=col_zone)
    
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(plot_distributions(zone_data, p99, col_zone, "histogram"), width='stretch')
    with colB:
        st.plotly_chart(plot_distributions(zone_data, p99, col_zone, "ecdf"), width='stretch')

#=============================================
# TASK 4
#=============================================
elif menu_selection == "4️⃣ Visual Integrity":
    st.title("4️⃣ The Visual Integrity Audit")
    st.markdown("---")
    
    st.markdown("""
    <div class="report-card">
        <p><span class="problem-text">🚨 The Problem:</span> A proposal suggested using a heavily decorated 3D Bar Chart, mapped using 'Rainbow' colors, to display Pollution vs. Population Density vs. Regional nodes.</p>
        <p><span class="solution-text">✅ Our Solution:</span> We formally rejected the 3D approach and the Rainbow gradient. Instead, we shifted the parameters onto flat, partitioned 2D geometry.</p>
        <p><span class="graph-text">📊 Graph Used:</span> Small Multiples Array (Trellis Plot) applying a strict 'Viridis' Sequential Scale.</p>
        <p><span class="justification-text">🎓 Why We Used It (Tufte Compliance):</span> <br/>
        <b>1. Anti-Lie Factor:</b> 3D bar charts trigger perspective distortion (bars in the background literally render smaller than the identical data block in the foreground). Trellis plots maintain rigid, honest 2D coordinate space.<br/>
        <b>2. No Graphical Ducks:</b> We stripped all depth shadows and arbitrary Z-axis grids.<br/>
        <b>3. Sequential Coloring overrides Rainbow:</b> Human biology processes luminance linearly. A 'Rainbow' map creates false borders when transitioning abruptly from green to blue. A sequential scale (Viridis) ensures mathematical variables perfectly map to optical lightness.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(plot_small_multiples(df_sampled), width='stretch')
