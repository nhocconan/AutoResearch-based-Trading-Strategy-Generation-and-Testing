#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d pivot level confirmation and volume spike
# Uses 6h Donchian channels for breakout direction, 1d Camarilla pivot levels (R3/S3) for
# institutional support/resistance, and volume spike (>2x 20-period average) for confirmation.
# Designed for medium frequency (target: 50-150 total trades over 4 years) with discrete
# position sizing to minimize fee drag. Works in trending markets via breakouts and
# in ranging markets via fade at extreme pivot levels.

name = "6h_donchian20_daily_pivot_volume_v5"
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
    
    # 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + (range_1d * 1.1 / 2)
    s3_1d = pivot_1d - (range_1d * 1.1 / 2)
    r4_1d = pivot_1d + (range_1d * 1.1)
    s4_1d = pivot_1d - (range_1d * 1.1)
    
    # Align 1d pivot levels to 6h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > highest_high[i] and volume_spike[i]
        bearish_breakout = close[i] < lowest_low[i] and volume_spike[i]
        
        # Fade conditions at extreme pivot levels
        near_r3 = abs(close[i] - r3_1d_aligned[i]) / close[i] < 0.005  # within 0.5%
        near_s3 = abs(close[i] - s3_1d_aligned[i]) / close[i] < 0.005  # within 0.5%
        at_r4 = close[i] >= r4_1d_aligned[i]
        at_s4 = close[i] <= s4_1d_aligned[i]
        
        # Long: bullish breakout OR fade from S3/S4
        if bullish_breakout or near_s3 or at_s4:
            signals[i] = 0.25
        # Short: bearish breakout OR fade from R3/R4
        elif bearish_breakout or near_r3 or at_r4:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals