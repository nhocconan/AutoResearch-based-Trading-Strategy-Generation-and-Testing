#!/usr/bin/env python3
"""
12h_ThreeMonthHigh_LowBreakout_Trend_Confirmation
Hypothesis: Breakout of 3-month (65-day) high/low with 1d EMA50 trend confirmation and volume filter.
Targets major trend continuations after consolidation, works in bull (breakout highs) and bear (breakdown lows).
Designed for low frequency (~15-25 trades/year) to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and 3-month high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 65-day (approx 3 months) high and low
    high_65d = pd.Series(df_1d['high'].values).rolling(window=65, min_periods=65).max().values
    low_65d = pd.Series(df_1d['low'].values).rolling(window=65, min_periods=65).min().values
    
    # Align to 12h timeframe (previous day's levels available at open)
    high_65d_aligned = align_htf_to_ltf(prices, df_1d, high_65d)
    low_65d_aligned = align_htf_to_ltf(prices, df_1d, low_65d)
    
    # Volume filter: require volume > 1.8x 30-period average to confirm breakout strength
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 30  # need 30 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_65d_aligned[i]) or 
            np.isnan(low_65d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 3-month high with uptrend and volume confirmation
            if (close[i] > high_65d_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 3-month low with downtrend and volume confirmation
            elif (close[i] < low_65d_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks back below 3-month low or trend fails
            if (close[i] < low_65d_aligned[i] or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above 3-month high or trend fails
            if (close[i] > high_65d_aligned[i] or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ThreeMonthHigh_LowBreakout_Trend_Confirmation"
timeframe = "12h"
leverage = 1.0