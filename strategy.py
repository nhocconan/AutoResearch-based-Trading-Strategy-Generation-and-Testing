#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_VolumeConfirm
Hypothesis: Weekly pivot levels act as strong support/resistance on 6h timeframe. 
Breakout above weekly R1 with Donchian(20) confirmation and 1d EMA50 trend filter 
goes long; breakdown below weekly S1 with Donchian(20) confirmation and 1d EMA50 
trend filter goes short. Volume spike confirms institutional participation. 
Designed for 6h to achieve 50-150 total trades over 4 years (12-37/year). 
Works in bull/bear markets by following 1d trend while using weekly pivots for 
key levels and Donchian for breakout validation.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly OHLC for pivot levels (prior week's values)
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Weekly pivot points: P = (H+L+C)/3
    weekly_p = (h_1w + l_1w + c_1w) / 3.0
    # Weekly R1 = (2*P) - L, S1 = (2*P) - H
    weekly_r1 = (2 * weekly_p) - l_1w
    weekly_s1 = (2 * weekly_p) - h_1w
    
    # Align weekly indicators to 6h timeframe (completed bars only)
    p_aligned = align_htf_to_ltf(prices, df_1w, weekly_p)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian channel (20-period) for breakout confirmation
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA50 (50) + Donchian (20) + volume avg (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(p_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        p_val = p_aligned[i]
        highest_20_val = highest_20[i]
        lowest_20_val = lowest_20[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Weekly R1/S1 breakout with Donchian confirmation and 1d EMA50 trend filter
            # Long: price closes above weekly R1 AND above Donchian high AND above 1d EMA50 (uptrend) AND volume spike
            long_condition = (close_val > r1_val) and (close_val > highest_20_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below weekly S1 AND below Donchian low AND below 1d EMA50 (downtrend) AND volume spike
            short_condition = (close_val < s1_val) and (close_val < lowest_20_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit conditions:
            # 1. Price touches weekly pivot (mean reversion to pivot)
            # 2. 1d EMA50 turns bearish (price below EMA)
            # 3. Price breaks below Donchian low (breakout failure)
            exit_condition = (close_val < p_val) or (close_val < ema_val) or (close_val < lowest_20_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. Price touches weekly pivot (mean reversion to pivot)
            # 2. 1d EMA50 turns bullish (price above EMA)
            # 3. Price breaks above Donchian high (breakout failure)
            exit_condition = (close_val > p_val) or (close_val > ema_val) or (close_val > highest_20_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0