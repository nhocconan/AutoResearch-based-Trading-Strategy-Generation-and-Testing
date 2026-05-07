#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_Trend_Filter
Hypothesis: For 6h timeframe, use weekly pivot points from 1w data as structural support/resistance.
Breakouts above weekly R1 or below weekly S1 with volume confirmation and trend filter (price > 6h EMA50 for longs, < 6h EMA50 for shorts) capture institutional flow.
Weekly pivots provide significant levels that work in both bull and bear markets as price remembers these levels.
Target: 15-25 trades per year (~60-100 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "6h_Weekly_Pivot_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    # Then R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_ltf_to_htf(prices, df_1w, weekly_pivot)
    r1_aligned = align_ltf_to_htf(prices, df_1w, weekly_r1)
    s1_aligned = align_ltf_to_htf(prices, df_1w, weekly_s1)
    
    # 6h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need 50 for EMA50 plus buffer
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend regime from 6h EMA50
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: break above weekly R1 in uptrend with volume
            long_breakout = close[i] > r1_aligned[i]
            # Short: break below weekly S1 in downtrend with volume
            short_breakout = close[i] < s1_aligned[i]
            
            if long_breakout and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif short_breakout and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below weekly pivot or trend changes to downtrend
            if close[i] < pivot_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above weekly pivot or trend changes to uptrend
            if close[i] > pivot_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def align_ltf_to_htf(prices, df_htf, htf_values):
    """Align higher timeframe values to lower timeframe index."""
    from mtf_data import align_htf_to_ltf
    return align_htf_to_ltf(prices, df_htf, htf_values)