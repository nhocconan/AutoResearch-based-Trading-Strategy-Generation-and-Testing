#!/usr/bin/env python3
name = "12h_Turtle_Soup_1dTrend_1wVolFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    # Daily 20-period high/low for Turtle Soup (false breakout fade)
    high_20d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Weekly volume filter (avoid low-volume weeks)
    vol_5w = pd.Series(df_1w['volume'].values).rolling(window=5, min_periods=5).mean().values
    
    # Align to 12h timeframe
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    vol_5w_aligned = align_htf_to_ltf(prices, df_1w, vol_5w)
    
    # Daily trend filter (20 EMA)
    close_1d = df_1d['close'].values
    ema_20d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20d_aligned = align_htf_to_ltf(prices, df_1d, ema_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for daily 20-period calculations
    
    for i in range(start_idx, n):
        if np.isnan(high_20d_aligned[i]) or np.isnan(low_20d_aligned[i]) or np.isnan(ema_20d_aligned[i]) or np.isnan(vol_5w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: only trade when weekly volume is above average
        vol_filter = volume[i] > vol_5w_aligned[i] * 0.8  # Allow some flexibility
        
        if position == 0:
            # Turtle Soup Long: false breakdown below 20-day low, then reversal
            # Condition: price breaks below 20-day low but closes back above it
            breakdown = low[i] < low_20d_aligned[i]
            recovery = close[i] > low_20d_aligned[i]
            
            # Only take long if in uptrend (price above daily 20 EMA)
            uptrend = close[i] > ema_20d_aligned[i]
            
            if breakdown and recovery and vol_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Turtle Soup Short: false breakout above 20-day high, then reversal
            # Condition: price breaks above 20-day high but closes back below it
            breakout = high[i] > high_20d_aligned[i]
            failure = close[i] < high_20d_aligned[i]
            
            # Only take short if in downtrend (price below daily 20 EMA)
            downtrend = close[i] < ema_20d_aligned[i]
            
            if breakout and failure and vol_filter and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks above 20-day high (trend continuation) or stops
            if high[i] > high_20d_aligned[i] or low[i] < low_20d_aligned[i] * 0.95:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks below 20-day low (trend continuation) or stops
            if low[i] < low_20d_aligned[i] or high[i] > high_20d_aligned[i] * 1.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Turtle Soup strategy with daily trend filter and weekly volume filter.
# Turtle Soup fades false breakouts of 20-day highs/lows - a proven mean reversion edge.
# In uptrends: buy breakdowns of 20-day low that reverse back above (long).
# In downtrends: sell breakouts of 20-day high that fail back below (short).
# Daily 20 EMA ensures trades align with intermediate trend.
# Weekly volume filter avoids low-volatility chop.
# Works in both bull (buy false breakdowns in uptrend) and bear (sell false breakouts in downtrend).
# Position size 0.25 keeps trades ~20-40/year to minimize fee drag.