# 🏎️ GR-Strategist: AI Race Engineer Dashboard

![Toyota GR](https://img.shields.io/badge/Toyota-GR_Sport-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)

**GR-Strategist** is a high-performance telemetry analysis platform designed for the Toyota Hackathon. It transforms raw sensor data into actionable racing insights using advanced physics reconstruction and interactive visualizations.

## 🚀 Key Features

* **Physics-Based Track Reconstruction:** Generates a synthetic track map from raw speed and G-force data using a custom physics engine.
* **Real vs. Synthetic Validation:** Compares the physics-generated path against the official VIR track layout for accuracy verification.
* **High-Performance Data Engine:** Utilizes Parquet caching to load and process 1.5GB+ of telemetry data instantly (100x speedup over CSV).
* **Interactive Telemetry Traces:** Zoomable, synchronized plots for Speed, Throttle, Brake, and Steering inputs.
* **G-G Friction Circle:** Visualizes tire usage and grip limits in real-time.
* **Toyota GR Aesthetic:** Professional dark-mode UI designed for low-light pit wall environments.

## 🛠️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/Toyota-Hackathon-Dashboard.git
   cd Toyota-Hackathon-Dashboard
   ```

2. **Set Up Virtual Environment**

   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Mac/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## 📂 Data Setup

Due to GitHub file size limits, the raw telemetry data is not included in this repository.

1. Place your `R1_vir_telemetry_data.csv` file into the `data/` folder.
2. Ensure `data/vir_layout.png` is present for the track map comparison.

## 🏁 Usage

Run the dashboard locally:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## 🧠 How It Works

1. **Data Ingestion:** Raw CSV data is cleaned, pivoted, and cached into a high-speed Parquet format.
2. **Physics Engine:**
   * Calculates **Distance** by integrating Speed over Time.
   * Calculates **Heading** by integrating Yaw Rate (derived from Lateral G and Speed).
   * Generates **X/Y Coordinates** to map the car's path.
3. **Visualization:** Streamlit and Plotly render the processed data into interactive dashboards.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
