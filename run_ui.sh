#!/bin/sh
# This script forces Streamlit to run on 8501, regardless of what Cloud Run thinks
streamlit run admin_ui.py --server.port=8501 --server.address=0.0.0.0