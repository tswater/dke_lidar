# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""
Cloud Cover Screening for Shallow Cumulus
"""

# %%
# Data Required: ARM COGS Data + GOES CMI Data (Red visible channel)
# ARM: from the ARM website
# GOES: earthdata.nasa.gov

# Condition: R(cloud) >= R(clear-sky) + R(threshold)
# Threshold = 0.045 (already validated by previous studies for the SGP)
# Evaluate at each pixel

# 
