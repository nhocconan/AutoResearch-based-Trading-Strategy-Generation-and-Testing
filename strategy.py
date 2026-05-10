#!/usr/bin/env python3
"""
1d_KAMA_With_WeeklyTrend_And_Volume
Hypothesis: Use daily KAMA direction as primary trend filter, combined with weekly
trend alignment (price vs weekly pivot) and daily volume confirmation.
KAMA adapts to market noise, reducing whipsaws in chop. Weekly pivot ensures
trades align with major trend. Volume filter avoids low-conviction breakouts.
Designed for 15-25 trades/year per symbol, works in bull/bear via dual trend filter.
"""

name = "1d_KAMA_With_WeeklyTrend_And_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_length=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(er_length, len(close)):
        if volatility[i-er_length:i+1].sum() > 0:
            er[i] = change[i-er_length:i+1].sum() / volatility[i-er_length:i+1].sum()
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama_out = np.zeros_like(close)
    kama_out[0] = close[0]
    for i in range(1, len(close)):
        kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
    
    return kama_out

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_wk = df_1w['high'].shift(1).values
    low_wk = df_1w['low'].shift(1).values
    close_wk = df_1w['close'].shift(1).values
    
    pivot_wk = (high_wk + low_wk + close_wk) / 3.0
    
    # Align weekly pivot to daily timeframe
    pivot_wk_aligned = align_htf_to_ltf(prices, df_1w, pivot_wk)
    
    # Get daily data for KAMA and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10,2,30)
    kama_val = kama(close, er_length=10, fast=2, slow=30)
    
    # Volume filter: current volume > 1.3x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30 for slow EMA) + weekly pivot
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_val[i]) or 
            np.isnan(pivot_wk_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend alignment
        # Daily trend: price vs KAMA
        daily_bullish = close[i] > kama_val[i]
        daily_bearish = close[i] < kama_val[i]
        
        # Weekly trend: price vs weekly pivot
        weekly_bullish = close[i] > pivot_wk_aligned[i]
        weekly_bearish = close[i] < pivot_wk_aligned[i]
        
        if position == 0:
            # Long: daily bullish AND weekly bullish AND volume
            if daily_bullish and weekly_bullish and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily bearish AND weekly bearish AND volume
            elif daily_bearish and weekly_bearish and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: daily bearish OR weekly bearish
            if not daily_bullish or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: daily bullish OR weekly bullish
            if not daily_bearish or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals