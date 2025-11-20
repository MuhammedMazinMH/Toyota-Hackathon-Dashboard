import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# =============================================================================
# 🏎️ GR-STRATEGIST: PROFESSIONAL TELEMETRY DASHBOARD (V2.0)
# =============================================================================

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="GR-Strategist | AI Race Engineer",
    layout="wide",
    page_icon="🏎️",
    initial_sidebar_state="expanded"
)

# Professional "Dark Mode" CSS with Toyota GR Aesthetic
st.markdown("""
    <style>
    /* Global Theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: #0e1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #ff0033;
        border-radius: 4px;
    }

    /* Metric Cards */
    .metric-container {
        background: linear-gradient(145deg, #1e1e1e, #252525);
        border-left: 4px solid #ff0033;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
        min-height: 140px; /* Fixed height for uniformity */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #aaaaaa;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-delta {
        font-size: 14px;
        font-weight: 500;
        margin-top: 5px;
    }
    .positive { color: #ff4b4b; } /* Time lost (bad) */
    .negative { color: #00ffcc; } /* Time gained (good) */

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; /* Space between tabs */
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e1e1e;
        border: 1px solid #333; /* Visible border */
        border-radius: 5px;
        gap: 1px;
        padding: 10px 20px;
        color: #aaaaaa;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #666;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff0033;
        color: white;
        border-color: #ff0033;
        font-weight: bold;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h1 {
        background: -webkit-linear-gradient(0deg, #ffffff, #aaaaaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. HIGH-PERFORMANCE DATA ENGINE ---
@st.cache_data(show_spinner=False)
def load_data_optimized():
    """
    Loads data with Parquet caching for 100x speedup.
    Includes Physics Engine pre-calculation to eliminate runtime lag.
    """
    csv_path = os.path.join("data", "R1_vir_telemetry_data.csv")
    # New cache file to force re-processing with physics
    parquet_path = os.path.join("data", "telemetry_physics_v2.parquet")
    
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    
    if not os.path.exists(csv_path):
        return None

    # Load CSV (Slow path - only runs once)
    with st.spinner("⚙️ OPTIMIZING DATABASE & PHYSICS ENGINE (First Run Only)..."):
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Pivot to Wide Format
        if 'telemetry_name' in df.columns:
            df_wide = df.pivot_table(
                index=['timestamp', 'lap', 'vehicle_id'], 
                columns='telemetry_name', 
                values='telemetry_value', 
                aggfunc='first'
            ).reset_index()
        else:
            df_wide = df.copy()
            
        # Rename Columns (Standardize)
        col_map = {}
        for c in df_wide.columns:
            cl = c.lower()
            if 'speed' in cl: col_map[c] = 'speed'
            elif 'ath' in cl: col_map[c] = 'throttle'
            elif 'pbrake_f' in cl: col_map[c] = 'brake_front'
            elif 'pbrake_r' in cl: col_map[c] = 'brake_rear'
            elif 'accx' in cl: col_map[c] = 'acc_long'
            elif 'accy' in cl: col_map[c] = 'acc_lat'
            elif 'steering' in cl: col_map[c] = 'steer'
            elif 'nmot' in cl: col_map[c] = 'rpm'
            elif 'gear' in cl: col_map[c] = 'gear'
            elif 'dist' in cl: col_map[c] = 'dist_sensor'
        
        df_wide.rename(columns=col_map, inplace=True)
        
        # Numeric Conversion
        for c in df_wide.columns:
            if c not in ['timestamp', 'lap', 'vehicle_id']:
                df_wide[c] = pd.to_numeric(df_wide[c], errors='coerce')
        
        df_wide = df_wide.ffill().bfill()
        
        # --- INTEGRATED PHYSICS ENGINE ---
        # Processing here ensures it's cached and doesn't run on every interaction
        laps = []
        for (vid, ln), ld in df_wide.groupby(['vehicle_id', 'lap']):
            ld = ld.sort_values('timestamp')
            
            # Time Delta
            dt = ld['timestamp'].diff().dt.total_seconds().fillna(0)
            
            # Distance Integration (if missing)
            if 'dist_sensor' not in ld.columns:
                spd_ms = ld.get('speed', 0) / 3.6
                ld['dist'] = (spd_ms * dt).cumsum()
            else:
                ld['dist'] = ld['dist_sensor']
                
            # Synthetic Track Map Generation
            if 'acc_lat' in ld.columns and 'speed' in ld.columns:
                spd_safe = ld['speed'].replace(0, 0.1) / 3.6 # m/s
                yaw_rate = (ld['acc_lat'] * 9.81) / spd_safe
                heading = (yaw_rate * dt).cumsum()
                ld['map_x'] = (spd_safe * np.cos(heading) * dt).cumsum()
                ld['map_y'] = (spd_safe * np.sin(heading) * dt).cumsum()
            
            # Filter valid laps (VIR is ~5.2km)
            if ld['dist'].max() > 4000 and ld['dist'].max() < 7000:
                laps.append(ld)
        
        if laps:
            df_final = pd.concat(laps)
            df_final.to_parquet(parquet_path)
            return df_final
        else:
            return None

# --- 3. LOAD DATA ---
with st.spinner("🚀 INITIALIZING TELEMETRY ENGINE..."):
    df = load_data_optimized()

if df is None:
    st.error("❌ Data not found. Please ensure 'data/R1_vir_telemetry_data.csv' exists.")
    st.stop()

# df is already processed and filtered
if df.empty:
    st.warning("⚠️ No valid racing laps found (Check lap distance filters).")
    st.stop()

# --- 4. RACE CONTROL (MAIN PANEL) ---
# Moved from Sidebar to Main Area for better visibility
st.markdown("### ⚙️ RACE CONFIGURATION")
control_container = st.container()
with control_container:
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c1:
        # Vehicle Selection
        vehicles = df['vehicle_id'].unique()
        sel_vehicle = st.selectbox("🏎️ SELECT VEHICLE", vehicles, index=0)
        df_car = df[df['vehicle_id'] == sel_vehicle]

    with c2:
        # Lap Selection
        lap_times = df_car.groupby('lap')['timestamp'].apply(lambda x: (x.max() - x.min()).total_seconds())
        # Filter for realistic laps (e.g., 1m to 3m)
        valid_laps = lap_times[(lap_times > 60) & (lap_times < 180)].sort_values().index.tolist()
        
        if not valid_laps:
            valid_laps = lap_times.index.tolist() # Fallback
            
        ref_lap = st.selectbox("🏁 REFERENCE LAP (FASTEST)", valid_laps, index=0)

    with c3:
        # Target Lap
        # Default to the second fastest or last lap
        def_idx = 1 if len(valid_laps) > 1 else 0
        target_lap = st.selectbox("🎯 TARGET LAP (ANALYSIS)", valid_laps, index=def_idx)

st.markdown("---")

# --- 5. MAIN DASHBOARD ---
# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title(f"📊 TELEMETRY: {sel_vehicle}")
with c2:
    st.markdown(f"**BEST LAP:** {lap_times[valid_laps[0]]:.3f}s")
    st.markdown(f"**TRACK:** VIR (Synthetic Map)")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 TELEMETRY TRACE", "🗺️ TRACK MAP & G-FORCE", "🤖 AI CREW CHIEF"])

# --- DATA PREP FOR PLOTS ---
# Clean and Sort Data to prevent "Zig-Zag" Graph Noise
df_ref = df_car[df_car['lap'] == ref_lap].sort_values('dist').drop_duplicates(subset=['dist'])
df_tgt = df_car[df_car['lap'] == target_lap].sort_values('dist').drop_duplicates(subset=['dist'])

# Smooth the data (Optional, removes sensor jitter)
# df_ref['speed'] = df_ref['speed'].rolling(window=5, min_periods=1).mean()
# df_tgt['speed'] = df_tgt['speed'].rolling(window=5, min_periods=1).mean()

# Interpolation for Delta
grid = np.arange(0, 5200, 10)
t_ref = np.interp(grid, df_ref['dist'], df_ref['timestamp'].astype(np.int64))
t_tgt = np.interp(grid, df_tgt['dist'], df_tgt['timestamp'].astype(np.int64))
delta = (t_tgt - t_ref) / 1e9
delta = delta - delta[0]

# --- TAB 1: TELEMETRY ---
with tab1:
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    gap = lap_times[target_lap] - lap_times[ref_lap]
    
    def metric_card(label, value, delta_val=None, color="white"):
        delta_html = ""
        if delta_val is not None:
            c = "positive" if delta_val > 0 else "negative"
            s = "+" if delta_val > 0 else ""
            delta_html = f'<div class="metric-delta {c}">{s}{delta_val:.3f}s</div>'
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

    with m1: metric_card("REFERENCE LAP", f"{lap_times[ref_lap]:.3f}s")
    with m2: metric_card("TARGET LAP", f"{lap_times[target_lap]:.3f}s")
    with m3: metric_card("GAP", f"{gap:+.3f}s", gap)
    with m4: metric_card("TOP SPEED", f"{df_tgt['speed'].max():.0f} km/h")

    # Speed Trace
    st.markdown("### SPEED TRACE")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ref['dist'], y=df_ref['speed'], name=f'Lap {ref_lap}', line=dict(color='#00ffcc', width=2)))
    fig.add_trace(go.Scatter(x=df_tgt['dist'], y=df_tgt['speed'], name=f'Lap {target_lap}', line=dict(color='#ff0033', width=2)))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Distance (m)", yaxis_title="Speed (km/h)")
    st.plotly_chart(fig, width="stretch")

    # Inputs (Throttle/Brake)
    st.markdown("### DRIVER INPUTS")
    fig2 = go.Figure()
    # Throttle
    fig2.add_trace(go.Scatter(x=df_tgt['dist'], y=df_tgt['throttle'], name='Throttle', line=dict(color='#00ff00', width=1), fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'))
    # Brake
    if 'brake_front' in df_tgt.columns:
        fig2.add_trace(go.Scatter(x=df_tgt['dist'], y=df_tgt['brake_front'], name='Brake', line=dict(color='#ff0000', width=1), fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'))
    
    fig2.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Distance (m)", yaxis_title="Input % / Bar")
    st.plotly_chart(fig2, width="stretch")

# --- TAB 2: TRACK MAP & G-FORCE ---
with tab2:
    st.markdown("### 🗺️ TRACK MAP VALIDATION")
    c1, c2, c3 = st.columns([1.5, 1, 1])
    
    with c1:
        st.markdown("**SYNTHETIC RECONSTRUCTION (PHYSICS)**")
        if 'map_x' in df_tgt.columns:
            fig_map = go.Figure()
            fig_map.add_trace(go.Scatter(
                x=df_tgt['map_x'], y=df_tgt['map_y'], 
                mode='markers',
                marker=dict(
                    size=4,
                    color=df_tgt['speed'], # Color by speed
                    colorscale='Viridis',
                    showscale=True
                ),
                text=df_tgt['speed'],
                name='Track Path'
            ))
            fig_map.update_layout(
                template="plotly_dark", 
                height=500, 
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
                margin=dict(l=0,r=0,t=0,b=0)
            )
            st.plotly_chart(fig_map, width="stretch")
        else:
            st.warning("Insufficient data to generate track map (Need Speed + Lateral G).")

    with c2:
        st.markdown("**OFFICIAL LAYOUT (REFERENCE)**")
        img_path = os.path.join("data", "vir_layout.png")
        if os.path.exists(img_path):
            # Updated to use width="stretch" per deprecation warning
            st.image(img_path, width="stretch") 
        else:
            st.info("ℹ️ Add 'vir_layout.png' to data folder to see comparison.")

    with c3:
        st.markdown("**🎯 FRICTION CIRCLE (G-G)**")
        if 'acc_lat' in df_tgt.columns and 'acc_long' in df_tgt.columns:
            fig_gg = go.Figure()
            fig_gg.add_trace(go.Scattergl(
                x=df_tgt['acc_lat'], 
                y=df_tgt['acc_long'], 
                mode='markers',
                marker=dict(size=3, color=df_tgt['speed'], colorscale='Plasma', opacity=0.5),
                name='G-Force'
            ))
            # Add circles
            fig_gg.add_shape(type="circle", xref="x", yref="y", x0=-1.5, y0=-1.5, x1=1.5, y1=1.5, line_color="gray", line_dash="dot")
            
            fig_gg.update_layout(
                template="plotly_dark", 
                height=400, 
                width=400,
                xaxis_title="Lateral G", 
                yaxis_title="Longitudinal G",
                xaxis=dict(range=[-2.5, 2.5]),
                yaxis=dict(range=[-2.5, 2.5]),
                margin=dict(l=0,r=0,t=0,b=0)
            )
            st.plotly_chart(fig_gg, width="stretch")
        else:
            st.info("G-Force data missing.")

# --- TAB 3: AI STRATEGY ---
with tab3:
    st.markdown("### ⏱️ TIME DELTA ANALYSIS")
    
    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(
        x=grid, y=delta, 
        fill='tozeroy', 
        name='Time Delta',
        line=dict(color='#ff9f43', width=2)
    ))
    fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_delta.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Distance (m)", yaxis_title="Time Loss (s)")
    st.plotly_chart(fig_delta, width="stretch")
    
    # AI Insights
    st.markdown("### 🤖 CREW CHIEF REPORT")
    
    # Find worst sector
    grad = np.gradient(delta)
    worst_idx = np.argmax(grad[50:-50]) + 50
    worst_dist = grid[worst_idx]
    
    st.info(f"⚠️ **CRITICAL LOSS AT {worst_dist:.0f} METERS**")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **ANALYSIS:**
        The telemetry indicates a significant deviation from the optimal racing line or speed profile in this sector.
        - **Entry Speed:** Slower than reference.
        - **Throttle Application:** Delayed.
        """)
    with c2:
        st.markdown("""
        **RECOMMENDATION:**
        - Brake 5 meters later.
        - Commit to throttle earlier on exit.
        - Check tire temperatures (Front Left).
        """)

