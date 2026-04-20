#!/usr/bin/env python3
"""
12h_Donchian20_WeeklyTrend_DailyVolume_Breakout
Concept: 12h Donchian(20) breakout with weekly trend filter and daily volume spike confirmation.
- Long: Price > Donchian high(20) AND weekly EMA(20) > EMA(50) AND daily volume > 1.5x 20-period average
- Short: Price < Donchian low(20) AND weekly EMA(20) < EMA(50) AND daily volume > 1.5x 20-period average
- Exit: Price crosses back through Donchian midpoint
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years
- Works in bull/bear: Weekly trend filter adapts, volume spike confirms institutional interest
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_WeeklyTrend_DailyVolume_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Weekly: EMA Trend Filter (20 and 50) ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMAs to 12h
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 12h: Donchian Channel (20-period) ===
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
        ema20 = ema_20_1w_aligned[i]
        ema50 = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(dh) or np.isnan(dl) or np.isnan(dm) or 
            np.isnan(ema20) or np.isnan(ema50) or np.isnan(vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-period average
        # Use the aligned daily volume value
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high with weekly uptrend and volume spike
            if close[i] > dh and ema20 > ema50 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with weekly downtrend and volume spike
            elif close[i] < dl and ema20 < ema50 and vol_condition:
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