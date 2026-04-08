#!/usr/bin/env python3
"""
12h_weekly_pivot_volume_v1
Hypothesis: Weekly pivot as institutional reference + volume spike for breakout.
- Trade only when price breaks above/below weekly pivot with volume confirmation
- Weekly trend filter: price above/below weekly 20-period EMA
- Exit on opposite pivot touch or weekly trend reversal
- Target: 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot
    high_1w_prev = df_1w['high'].shift(1).values
    low_1w_prev = df_1w['low'].shift(1).values
    close_1w_prev = df_1w['close'].shift(1).values
    
    # Weekly pivot point
    weekly_pivot = (high_1w_prev + low_1w_prev + close_1w_prev) / 3
    
    # Weekly trend: 20-period EMA
    weekly_close_series = pd.Series(df_1w['close'])
    weekly_ema20 = weekly_close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_bullish = df_1w['close'].values > weekly_ema20
    weekly_bearish = df_1w['close'].values < weekly_ema20
    
    # Align weekly data to 12h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume spike: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(volume_ma20[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price touches weekly pivot from above OR weekly turns bearish
            if low[i] <= weekly_pivot_aligned[i] or weekly_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price touches weekly pivot from below OR weekly turns bullish
            if high[i] >= weekly_pivot_aligned[i] or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: weekly bullish + price breaks above pivot with volume spike
            if (weekly_bullish_aligned[i] > 0.5 and 
                high[i] > weekly_pivot_aligned[i] and volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: weekly bearish + price breaks below pivot with volume spike
            elif (weekly_bearish_aligned[i] > 0.5 and 
                  low[i] < weekly_pivot_aligned[i] and volume_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals