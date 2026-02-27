🌍 Urban Environmental Intelligence Challenge

This project presents a production-grade Smart City Environmental Diagnostic Engine built using 2025 OpenAQ global air quality data from 100 sensor stations. The system is designed to detect environmental anomalies, analyze pollution drivers, and ensure visual integrity in data storytelling.

The engine applies dimensionality reduction, high-density temporal modeling, and heavy-tail distribution analysis while adhering strictly to Data-Ink Ratio and Lie Factor principles.

🎯 Key Objectives

Analyze six environmental variables (PM2.5, PM10, NO₂, Ozone, Temperature, Humidity)

Identify clustering patterns between Industrial and Residential zones

Detect synchronized health threshold violations (PM2.5 > 35 µg/m³)

Estimate extreme hazard probability (PM2.5 > 200 µg/m³)

Ensure visual honesty in all analytical outputs

Deploy scalable analytics pipeline with reproducibility

🧠 Analytical Techniques Used

PCA (Principal Component Analysis) after standardization

Hourly temporal matrix analysis (100 × 8760 structure)

High-density heatmap visualization

ECDF for tail integrity

99th percentile estimation

Small multiples visualization for multi-variable comparison

Sequential color scale selection for perceptual accuracy

⚙️ Technical Features

Modular Python pipeline (.py files only)

Big Data optimized using columnar storage (Parquet)

Lazy data loading

Streamlit interactive dashboard

API ingestion from OpenAQ

No 3D distortions or graphical clutter

Reproducible and deployment-ready

📊 Deployment

Interactive Dashboard:
🔗 [Your Streamlit Link Here]

🏗️ Architecture

Data Ingestion → Data Cleaning → Feature Standardization → PCA →
Temporal Analysis → Distribution Modeling → Visualization Engine → Streamlit Deployment

📌 Academic Focus

This project emphasizes:

Data-Ink Ratio optimization

Avoidance of scale distortion

Tail honesty in heavy-tailed distributions

Perceptual correctness in color design

Reproducibility and modular design
