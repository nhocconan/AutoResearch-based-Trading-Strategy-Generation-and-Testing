#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels for structure and 1d volume spikes for confirmation.
# In weekly uptrend (price > weekly pivot), go long when price breaks above R3 with 1d volume spike.
# In weekly downtrend (price < weekly pivot), go short when price breaks below S3 with 1d volume spike.
# Weekly trend filter reduces whipsaw; volume spike confirms institutional participation.
# Designed for low trade frequency (12-30/year) to minimize fee drag while capturing sustained moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w and 1d HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly Indicators: Camarilla Pivot Points ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on prior week)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1 / 2
    s3_1w = pivot_1w - range_1w * 1.1 / 2
    r4_1w = pivot_1w + range_1w * 1.1
    s4_1w = pivot_1w - range_1w * 1.1
    
    # Align weekly levels to 6h (wait for weekly bar close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 1d Indicators: Volume Spike (2.0x 20-period SMA) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_sma_20 * 2.0)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Weekly uptrend (price > weekly pivot)
        # 2. Price breaks above R3 (continuation breakout)
        # 3. 1d volume spike (institutional participation)
        if (close[i] > pivot_1w_aligned[i]) and (close[i] > r3_1w_aligned[i]) and vol_spike[i]:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Weekly downtrend (price < weekly pivot)
        # 2. Price breaks below S3 (continuation breakdown)
        # 3. 1d volume spike (institutional participation)
        elif (close[i] < pivot_1w_aligned[i]) and (close[i] < s3_1w_aligned[i]) and vol_spike[i]:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0