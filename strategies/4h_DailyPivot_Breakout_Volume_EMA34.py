# 4h_DailyPivot_Breakout_Volume_EMA34
# Hypothesis: Daily pivot levels provide strong support/resistance. Price breaking above R1 or below S1 with volume confirmation and trend alignment (EMA34) captures institutional breakouts. Works in both bull and bear markets as pivots adapt to volatility. Target: 20-40 trades/year on 4h timeframe to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe (use previous day's levels)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume filter: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need sufficient data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price relative to EMA34
        price_above_ema = close[i] > ema34_4h_aligned[i]
        price_below_ema = close[i] < ema34_4h_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and above EMA34
            if (close[i] > r1_4h[i] and volume_filter and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and below EMA34
            elif (close[i] < s1_4h[i] and volume_filter and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot or EMA34
            if close[i] < pivot_4h[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above pivot or EMA34
            if close[i] > pivot_4h[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_Volume_EMA34"
timeframe = "4h"
leverage = 1.0