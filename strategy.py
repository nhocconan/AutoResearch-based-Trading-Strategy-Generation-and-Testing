#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX momentum with 1d volume spike and choppiness regime filter
# - Long when TRIX(12) crosses above zero on 12h with 1d volume spike and choppy market (CHOP > 61.8)
# - Short when TRIX(12) crosses below zero on 12h with 1d volume spike and choppy market (CHOP > 61.8)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Volume spike and chop regime reduce false signals in trending markets
# - Works in both bull and bear markets by capturing mean reversion in choppy conditions

name = "12h_1d_trix_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(34) for TRIX calculation (typical period)
    ema1_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema2_1d = pd.Series(ema1_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema3_1d = pd.Series(ema2_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trix_1d = 100 * (ema3_1d - np.roll(ema3_1d, 1)) / np.roll(ema3_1d, 1)
    trix_1d[0] = 0  # First value undefined due to roll
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Choppiness Index (CHOP) - regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low))) 
    # Simplified: CHOP > 61.8 = ranging market (good for mean reversion)
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(tr1, np.abs(low_1d - np.roll(close_1d, 1)))
    tr2[0] = 0  # First TR undefined
    atr_14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(14) * (highest_high_14 - lowest_low_14)
    chop_denom = np.where(chop_denom == 0, 1, chop_denom)  # Avoid division by zero
    chop_1d = 100 * np.log10(sum_atr_14 / chop_denom)
    chop_1d = np.where(np.isnan(chop_1d), 50, chop_1d)  # Default to middle range
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when TRIX crosses below zero or volume spike ends
            if trix_1d_aligned[i] < 0 or not vol_spike_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when TRIX crosses above zero or volume spike ends
            if trix_1d_aligned[i] > 0 or not vol_spike_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only trade in choppy/range markets (CHOP > 61.8)
            if chop_1d_aligned[i] > 61.8:
                # Long signal: TRIX crosses above zero with volume spike
                if trix_1d_aligned[i] > 0 and trix_1d_aligned[i-1] <= 0 and vol_spike_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short signal: TRIX crosses below zero with volume spike
                elif trix_1d_aligned[i] < 0 and trix_1d_aligned[i-1] >= 0 and vol_spike_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals