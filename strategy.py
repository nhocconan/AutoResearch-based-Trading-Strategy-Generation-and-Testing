#!/usr/bin/env python3
"""
1d_1w_Donchian20_WeeklyTrend_DailyVolume_Breakout_v1
Concept: Daily price breaks weekly Donchian(20) with daily volume spike and weekly trend filter.
- Long: Close > weekly Donchian Upper(20) AND daily volume > 2.0x 20-period avg AND weekly close > weekly SMA(50)
- Short: Close < weekly Donchian Lower(20) AND daily volume > 2.0x 20-period avg AND weekly close < weekly SMA(50)
- Exit: Close crosses back through weekly midline
- Position sizing: 0.25
- Target: 30-100 total trades over 4 years (7-25/year)
- Works in bull/bear: weekly trend filter adapts, volume confirms institutional interest
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian20_WeeklyTrend_DailyVolume_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly: Donchian Channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # === Weekly: SMA(50) for trend filter ===
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # === Daily: Volume MA (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1d: Price ===
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for weekly SMA(50)
    
    for i in range(start_idx, n):
        # Get values
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        mid_val = donchian_mid_aligned[i]
        sma_50_val = sma_50_1w_aligned[i]
        vol_ma_20 = vol_ma_20_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(mid_val) or 
            np.isnan(sma_50_val) or np.isnan(vol_ma_20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 2.0x 20-period average
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 2.0 * vol_ma_20
        
        # Trend filter: weekly close above/below weekly SMA(50)
        weekly_close = close_1w[-1] if len(close_1w) > 0 else np.nan
        # Get aligned weekly close for current day
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        weekly_close_val = weekly_close_aligned[i]
        weekly_uptrend = weekly_close_val > sma_50_val
        weekly_downtrend = weekly_close_val < sma_50_val
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper with volume spike and weekly uptrend
            if close[i] > upper_val and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower with volume spike and weekly downtrend
            elif close[i] < lower_val and vol_condition and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly midline
            if close[i] < mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly midline
            if close[i] > mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals