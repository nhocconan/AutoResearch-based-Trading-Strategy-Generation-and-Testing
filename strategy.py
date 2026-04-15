#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike filter for 1h entries
# Long when price breaks above 4h Camarilla R3 level + 1d volume > 2x 20-period avg
# Short when price breaks below 4h Camarilla S3 level + 1d volume > 2x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee churn. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (15-35/year). Camarilla levels provide adaptive support/resistance
# based on prior day's range, effective in both ranging and trending markets. Volume filter ensures
# breakouts have conviction, reducing false signals in chop.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # === 4h Indicator: Camarilla Pivots (based on prior 4h bar) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    camarilla_r3 = np.full_like(close_4h, np.nan)
    camarilla_s3 = np.full_like(close_4h, np.nan)
    
    for i in range(1, len(df_4h)):
        # Prior 4h bar's range
        range_4h = high_4h[i-1] - low_4h[i-1]
        if range_4h <= 0:
            camarilla_r3[i] = camarilla_s3[i] = close_4h[i-1]
        else:
            camarilla_r3[i] = close_4h[i-1] + range_4h * 1.1 / 4
            camarilla_s3[i] = close_4h[i-1] - range_4h * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (already delayed by 1 bar in calculation)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # === 1d Indicator: Volume Spike Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # 20-period volume SMA on daily
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Volume spike: current daily volume > 2x 20-period average
    vol_spike_1d = volume_1d > (vol_sma_20_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 21  # Need at least 2 prior 4h bars for Camarilla + 20 for 1d vol SMA
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume must be spiking on 1d timeframe
        if vol_spike_1d_aligned[i] < 0.5:  # Not a spike
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # Price breaks above 4h Camarilla R3 level + 1d volume spike
        if close[i] > camarilla_r3_aligned[i]:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # Price breaks below 4h Camarilla S3 level + 1d volume spike
        elif close[i] < camarilla_s3_aligned[i]:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0