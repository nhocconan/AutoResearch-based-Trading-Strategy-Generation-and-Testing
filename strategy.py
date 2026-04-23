#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume spike confirmation.
Long when price breaks above 20-day high AND close > weekly EMA50 (uptrend) AND volume > 2.0x 20-day volume MA.
Short when price breaks below 20-day low AND close < weekly EMA50 (downtrend) AND volume > 2.0x 20-day volume MA.
Exit when price returns to the 10-day Donchian midpoint or opposite extreme is hit.
Designed for ~10-20 trades/year with structure-based edge that works in both bull and bear markets via trend filter.
Weekly EMA50 provides higher timeframe alignment to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    # We need to resample to daily, but we'll use the 1d HTF data for consistency
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20) on daily: upper = 20-day high, lower = 20-day low
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe first, then to 4h? No - we're trading on 1d timeframe
    # Since timeframe is 1d, we need to align HTF (1w) to 1d, and use 1d data directly
    # But our prices DataFrame is at 4h? Wait, timeframe=1d means we expect daily bars
    
    # Actually, let's check: the strategy says timeframe="1d", so prices DataFrame should have 1d bars
    # But the experiment history shows 4h strategies. Let me re-read...
    
    # Correction: The experiment says PRIMARY = 1d, HTF = 1w
    # So timeframe should be "1d", meaning we expect daily data in prices
    # However, the provided prices might be at 4h? No, the timeframe parameter tells the system
    # what resolution to expect. If timeframe="1d", prices will be daily bars.
    
    # But looking at the current strategy.py, it uses timeframe="4h" and gets 1d HTF data
    # So for timeframe="1d", we would get 1w HTF data and use 1d prices directly
    
    # Let's implement for timeframe="1d" as requested
    
    # Recalculate for 1d timeframe
    # prices DataFrame now contains daily bars
    
    # Calculate weekly EMA50 for trend filter (from 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian(20) from prices (which are daily)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_ma_20 + low_ma_20) / 2.0  # 10-day midpoint for exit
    
    # Volume confirmation: 20-day volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > weekly EMA50 = uptrend, close < weekly EMA50 = downtrend
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: daily volume > 2.0x 20-day volume MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_ma_20[i]  # Break above 20-day high
        breakout_down = close[i] < low_ma_20[i]  # Break below 20-day low
        return_to_mid = abs(close[i] - donchian_mid[i]) < (high_ma_20[i] - low_ma_20[i]) * 0.1  # Within 10% of midpoint
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above 20-day high AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-day low AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to midpoint or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = return_to_mid or opposite_extreme
            elif position == -1:
                exit_signal = return_to_mid or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0