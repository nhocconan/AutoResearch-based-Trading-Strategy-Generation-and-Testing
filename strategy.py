#!/usr/bin/env python3
"""
1h_4h1d_RangeBreakout_Volume_Trend_v1
Concept: Use 4h/1d to define trend and range, 1h for entry timing.
- Long: Price > 4h high AND price > 1d EMA50 AND 1h volume > 1.5x 20-period avg
- Short: Price < 4h low AND price < 1d EMA50 AND 1h volume > 1.5x 20-period avg
- Exit: Price crosses back below/above 4h mid
- Position sizing: 0.20
- Target: 60-150 total trades over 4 years (15-37/year)
- Works in bull/bear: 1d EMA50 trend filter adapts, volume confirms breakout strength
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_RangeBreakout_Volume_Trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h: Price Range (High/Low) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    range_high_4h = pd.Series(high_4h).rolling(window=1, min_periods=1).max().values  # current bar high
    range_low_4h = pd.Series(low_4h).rolling(window=1, min_periods=1).min().values    # current bar low
    range_mid_4h = (range_high_4h + range_low_4h) / 2.0
    
    range_high_4h_aligned = align_htf_to_ltf(prices, df_4h, range_high_4h)
    range_low_4h_aligned = align_htf_to_ltf(prices, df_4h, range_low_4h)
    range_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, range_mid_4h)
    
    # === 1d: EMA Trend Filter (50-period) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h: Volume Spike Filter ===
    volume_1h = prices['volume'].values
    vol_ma_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Price arrays
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        r_high = range_high_4h_aligned[i]
        r_low = range_low_4h_aligned[i]
        r_mid = range_mid_4h_aligned[i]
        ema50 = ema_50_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r_high) or np.isnan(r_low) or np.isnan(r_mid) or 
            np.isnan(ema50)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 1h volume > 1.5x 20-period average
        vol_condition = volume_1h[i] > 1.5 * vol_ma_20_1h[i]
        
        if position == 0:
            # Long: price breaks above 4h high with uptrend and volume spike
            if close[i] > r_high and close[i] > ema50 and vol_condition:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h low with downtrend and volume spike
            elif close[i] < r_low and close[i] < ema50 and vol_condition:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 4h mid
            if close[i] < r_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h mid
            if close[i] > r_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals