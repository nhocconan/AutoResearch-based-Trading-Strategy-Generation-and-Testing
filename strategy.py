#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels from 1w and 1d trend filter.
# Long when price is above 1w pivot + 1d close > EMA50.
# Short when price is below 1w pivot + 1d close < EMA50.
# Exit when price crosses back over 1w pivot.
# Uses 1w for structural bias (pivot), 6h for entry/execution, 1d for trend confirmation.
# Target: 20-50 trades/year to avoid fee drag, with strong directional bias.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot (based on previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # Using previous week's data to avoid look-ahead
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot using previous week's values (shift by 1 to avoid look-ahead)
    pivot_1w = np.full_like(high_1w, np.nan)
    for i in range(1, len(high_1w)):
        pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need at least 1 week of pivot data and 50-period EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(pivot_aligned[i]) or np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above weekly pivot with bullish 1d trend
            if price > pivot_aligned[i] and price > ema50_aligned[i]:
                signals[i] = size
                position = 1
            # Short: price below weekly pivot with bearish 1d trend
            elif price < pivot_aligned[i] and price < ema50_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below weekly pivot
            if price < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above weekly pivot
            if price > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_1dTrend"
timeframe = "6h"
leverage = 1.0