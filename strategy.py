#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_VolumeConfirm
Hypothesis: Combine weekly pivot points as structural support/resistance with 6h Donchian(20) breakouts for precise entry timing. Uses 1d EMA50 for trend filter and volume spike for confirmation. Weekly pivots provide multi-day structure that works in both bull and bear markets by identifying key institutional levels. Designed for low trade frequency (~15-30/year) with discrete position sizing (0.25) to minimize fee drag and improve test generalization. The weekly pivot filter reduces false breakouts while the Donchian breakout captures momentum when price breaks key weekly levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly OHLC for pivot points
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Weekly pivot points: P = (H+L+C)/3
    weekly_pivot = (h_1w + l_1w + c_1w) / 3.0
    # Weekly R1/S1: R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - l_1w
    weekly_s1 = 2 * weekly_pivot - h_1w
    # Weekly R2/S2: R2 = P + (H - L), S2 = P - (H - L)
    weekly_r2 = weekly_pivot + (h_1w - l_1w)
    weekly_s2 = weekly_pivot - (h_1w - l_1w)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and 1d indicators to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) on 6h: upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    
    # Warmup: need weekly pivot (1), Donchian (20), EMA50 (50), volume avg (20)
    start_idx = max(1, 20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_50_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly pivot and 1d EMA50 filter
            # Long: price breaks above Donchian upper AND above weekly R1 AND above 1d EMA50
            long_condition = (close_val > upper_val) and (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price breaks below Donchian lower AND below weekly S1 AND below 1d EMA50
            short_condition = (close_val < lower_val) and (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions:
            # 1. Price touches weekly S1 (opposite pivot level)
            # 2. 1d EMA50 turns bearish (price below EMA)
            # 3. Donchian lower breaks (momentum loss)
            exit_condition = (close_val < s1_val) or (close_val < ema_val) or (close_val < lower_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. Price touches weekly R1 (opposite pivot level)
            # 2. 1d EMA50 turns bullish (price above EMA)
            # 3. Donchian upper breaks (momentum loss)
            exit_condition = (close_val > r1_val) or (close_val > ema_val) or (close_val > upper_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0