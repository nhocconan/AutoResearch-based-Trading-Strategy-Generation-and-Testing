#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d pivot filter and volume confirmation
# Uses 6h Donchian breakout for trend direction, filtered by 1d Camarilla pivot levels
# and volume spike. Only trades breakouts in the direction of pivot bias.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear via fade at extremes.

name = "6h_donchian20_1d_pivot_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily Pivot points (using previous day's OHLC)
    # We need previous day's data, so shift by 1
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    r4 = prev_high + 3 * (pivot - prev_low)
    s4 = prev_low - 3 * (prev_high - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Donchian breakout above R3 with volume spike
        # or breakout above R4 (strong breakout)
        breakout_long = (close[i] > donchian_high[i-1]) and volume_spike[i]
        strong_breakout_long = close[i] > r4_aligned[i]
        
        # Short conditions: Donchian breakdown below S3 with volume spike
        # or breakdown below S4 (strong breakdown)
        breakout_short = (close[i] < donchian_low[i-1]) and volume_spike[i]
        strong_breakout_short = close[i] < s4_aligned[i]
        
        # Execute signals
        if breakout_long or strong_breakout_long:
            signals[i] = 0.25
        elif breakout_short or strong_breakout_short:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals