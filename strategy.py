#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: On 1d timeframe, breakouts above Camarilla R3 or below S3 with weekly trend alignment (EMA50) and volume spike generate persistent edges. Weekly trend filter avoids counter-trend trades in bear markets (2022) while capturing momentum in bull regimes. Volume confirmation reduces false breakouts. Designed for low trade frequency (~15-25/year) to minimize fee drag and improve generalization to 2025+ bearish/ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Load daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation
    PP = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    R3 = PP + range_1d * 1.1 / 4.0
    S3 = PP - range_1d * 1.1 / 4.0
    R4 = PP + range_1d * 1.1 / 2.0
    S4 = PP - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 1d timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike: current volume > 2.5 * 20-day average (stricter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for weekly EMA50 and daily pivots
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_trend_up = close_val > ema50_1w_aligned[i]
        weekly_trend_down = close_val < ema50_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: breakout with trend and volume
            long_breakout = (close_val > R3_aligned[i]) and weekly_trend_up and vol_spike
            short_breakout = (close_val < S3_aligned[i]) and weekly_trend_down and vol_spike
            
            if long_breakout:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit: weekly trend reversal or mean reversion at R4
            if not weekly_trend_up or close_val > R4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit: weekly trend reversal or mean reversion at S4
            if not weekly_trend_down or close_val < S4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0