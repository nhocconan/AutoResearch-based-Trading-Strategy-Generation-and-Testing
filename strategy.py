#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_VolumeConfirm
Hypothesis: Weekly pivot + Donchian(20) breakout with volume confirmation on 6h timeframe.
Weekly pivot provides institutional support/resistance levels. Donchian breakout confirms momentum.
In bull markets: price breaks above weekly R1 + Donchian(20) high + volume spike → long.
In bear markets: price breaks below weekly S1 + Donchian(20) low + volume spike → short.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
Requires volume confirmation (>1.5x average) to avoid false breakouts. Weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_val = ema_50_1d_aligned[i]
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(donch_high) or np.isnan(donch_low) or 
            np.isnan(ema_val) or np.isnan(avg_vol) or
            np.isnan(pivot) or np.isnan(r1) or np.isnan(s1)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above weekly R1 AND Donchian high with 1d uptrend and volume
        long_condition = (close_val > r1) and (close_val > donch_high) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below weekly S1 AND Donchian low with 1d downtrend and volume
        short_condition = (close_val < s1) and (close_val < donch_low) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or price returns to weekly pivot
        exit_long = (close_val < ema_val) or (close_val < pivot)
        exit_short = (close_val > ema_val) or (close_val > pivot)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0