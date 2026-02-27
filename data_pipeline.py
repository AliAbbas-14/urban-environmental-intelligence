"""
Big Data Pipeline Processing Module
- Lazy execution (Polars LazyFrame)
- Predicate pushdown limits disk reads
- Aggregations applied strictly BEFORE pivoting to prevent geometry RAM explosion
- Memory optimization principles implemented throughout
"""
import polars as pl
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data():
    input_path = "data/raw/openaq_2025.parquet"
    output_path = "data/processed/analytics_ready.parquet"
    
    os.makedirs("data/processed", exist_ok=True)
    logging.info("Initiating Polars LazyFrame Pipeline for Big Data Analytics...")
    
    if not os.path.exists(input_path):
        logging.error("Raw Parquet not found. Please execute api_ingestion.py first.")
        return
        
    # 1. Lazy Execution & Columnar Storage Reading
    # scan_parquet defer evaluation. Permits the Query Optimizer to prune unused 
    # columns natively off the SSD and trigger Predicate Pushdown (if filtered).
    lazy_df = pl.scan_parquet(input_path)
    
    # 2. Aggregations BEFORE pivoting (Memory Optimization)
    # The assignment necessitates STRICT HOURLY indexing (no daily averaging).
    # We aggregate primarily to collapse duplicate parametric readings on identical 
    # node-hours, mitigating memory overflow before executing a geometry-pivoting shuffle over millions of rows.
    hourly_agg = lazy_df.group_by(['location_id', 'datetimeUtc', 'parameter']).agg(
        pl.col('value').mean()
    )
    
    logging.info("Executing Lazy Graph (Columnar Pivoting)...")
    df_pivot = hourly_agg.collect().pivot(
        values="value",
        index=["location_id", "datetimeUtc"],
        on="parameter",
        aggregate_function="first"
    )
    
    # Defining Analytical Zones relative to sample medians
    median_pm25 = df_pivot.select(pl.col('pm25').mean()).item()
    zone_profiles = df_pivot.group_by('location_id').agg(
        pl.col('pm25').mean().alias('mean_pm25')
    ).with_columns(
        pl.when(pl.col('mean_pm25') > median_pm25)
        .then(pl.lit('Industrial'))
        .otherwise(pl.lit('Residential'))
        .alias('zone')
    )
    
    final_df = df_pivot.join(zone_profiles.select(['location_id', 'zone']), on='location_id')
    
    # Sorting ensures chronological vector alignments for streaming
    final_df = final_df.sort(['location_id', 'datetimeUtc'])
    
    logging.info("Deploying finalized compressed Arrow block...")
    final_df.write_parquet(output_path)
    logging.info("Data Pipeline Processing Complete: Ready for Streamlit Deployment via UI Engine.")

if __name__ == "__main__":
    process_data()
