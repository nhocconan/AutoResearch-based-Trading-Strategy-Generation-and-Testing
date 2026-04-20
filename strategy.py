#!/usr/bin/env python3
"""
12h_WeeklyTrend_DailyVolume_Breakout_v1
Concept: 12h price breaks above/below weekly Donchian channel with daily volume confirmation and trend filter.
- Long: Price > weekly Donchian high(20) AND daily volume > 1.5x 20-period avg AND daily EMA(50) > EMA(200)
- Short: Price < weekly Donchian low(20) AND daily volume > 1.5x 20-period avg AND daily EMA(50) < EMA(200)
- Exit: Price crosses back through weekly Donchian midpoint
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear: daily EMA trend filter adapts, volume confirms institutional interest
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_WeeklyTrend_DailyVolume_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === Weekly: Donchian Channel (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2.0
    
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    donchian_mid_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    
    # === Daily: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === Daily: EMA Trend Filter (50 and 200) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Price arrays
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        d_high = donchian_high_1w_aligned[i]
        d_low = donchian_low_1w_aligned[i]
        d_mid = donchian_mid_1w_aligned[i]
        ema50 = ema_50_1d_aligned[i]
        ema200 = ema_200_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(d_high) or np.isnan(d_low) or np.isnan(d_mid) or 
            np.isnan(ema50) or np.isnan(ema200)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-period average
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with uptrend and volume spike
            if close[i] > d_high and ema50 > ema200 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with downtrend and volume spike
            elif close[i] < d_low and ema50 < ema200 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly Donchian midpoint
            if close[i] < d_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly Donchian midpoint
            if close[i] > d_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals