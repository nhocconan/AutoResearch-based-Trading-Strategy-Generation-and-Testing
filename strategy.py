#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d Volume Spike + ADX Trend Filter
# Williams Alligator identifies trend direction via smoothed medians (Jaw/Teeth/Lips).
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish).
# Volume spike confirms breakout strength. ADX > 25 filters for trending markets.
# Works in bull markets (catch uptrends) and bear markets (catch downtrends).
# Target: 50-150 total trades over 4 years.
# Timeframe: 4h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for volume spike and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 4h (13,8,5 SMAs of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean()
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean()
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean()
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # 1d volume spike (current vs 20-period average)
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d / (vol_ma_20 + 1e-10)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(close_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(close_1d, 1)), 
                        np.maximum(np.roll(close_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(jaw_vals[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Bullish Alligator alignment: Lips > Teeth > Jaw
        bullish = lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i]
        # Bearish Alligator alignment: Lips < Teeth < Jaw
        bearish = lips_vals[i] < teeth_vals[i] and teeth_vals[i] < jaw_vals[i]
        
        # Long entry: Bullish alignment + volume spike + ADX > 25
        if bullish and vol_spike_aligned[i] > 2.0 and adx_aligned[i] > 25 and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: Bearish alignment + volume spike + ADX > 25
        elif bearish and vol_spike_aligned[i] > 2.0 and adx_aligned[i] > 25 and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite Alligator alignment or ADX < 20 (ranging market)
        elif position == 1 and (bearish or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dVolumeSpike_ADX"
timeframe = "4h"
leverage = 1.0