#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume confirmation
# Long when price breaks above 6h Camarilla R3 level + 1d volume > 1.5x 20-period avg
# Short when price breaks below 6h Camarilla S3 level + 1d volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years.
# Camarilla pivots provide mathematically derived support/resistance levels that work across market regimes.
# Volume confirmation ensures breakouts have institutional participation, reducing false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    vol_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    # === 6h Indicator: Camarilla Pivot Levels (based on previous day) ===
    # Calculate daily pivot points from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Classic pivot point formula
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R3 = pivot + (range_1d * 1.1 / 4.0)
    S3 = pivot - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20 + 1  # volume SMA + 1 for alignment
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d volume SMA (aligned)
        vol_confirm = volume[i] > (vol_sma_20_aligned[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # Price breaks above Camarilla R3 level + volume confirmation
        if (close[i] > R3_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below Camarilla S3 level + volume confirmation
        elif (close[i] < S3_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_Volume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0