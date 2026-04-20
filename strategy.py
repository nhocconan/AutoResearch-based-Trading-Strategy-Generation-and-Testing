#!/usr/bin/env python3
"""
6h_Donchian20_12hTrend_1dVolume_Breakout
Concept: Donchian(20) breakout with 12h trend filter and 1d volume spike confirmation.
- Long: Price > Donchian high(20) AND 12h EMA(50) > EMA(100) AND 1d volume > 1.5x 20-period average
- Short: Price < Donchian low(20) AND 12h EMA(50) < EMA(100) AND 1d volume > 1.5x 20-period average
- Exit: Price crosses back through Donchian midpoint
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years
- Works in bull/bear: Trend filter adapts, volume spike confirms institutional interest
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_12hTrend_1dVolume_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === 12h: EMA Trend Filter (50 and 100) ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 12h EMAs to 6h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    
    # === 1d: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 6h: Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian high/low using rolling window
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        ema50 = ema_50_12h_aligned[i]
        ema100 = ema_100_12h_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        vol = df_1d['volume'].values[i // 24] if i // 24 < len(df_1d) else vol_ma  # Simplified 1d volume access
        # Actually use the aligned volume MA
        vol = vol_ma  # Use the aligned MA directly
        
        # Skip if any value is NaN
        if (np.isnan(dh) or np.isnan(dl) or np.isnan(dm) or 
            np.isnan(ema50) or np.isnan(ema100) or np.isnan(vol)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        # Use the actual 1d volume value aligned
        vol_1d_val = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_val)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 1.5 * vol
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend and volume spike
            if close[i] > dh and ema50 > ema100 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with downtrend and volume spike
            elif close[i] < dl and ema50 < ema100 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint
            if close[i] < dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint
            if close[i] > dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals