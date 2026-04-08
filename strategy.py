#!/usr/bin/env python3
"""
6h_1w_1d_breakout_with_reversion_zone_v1
Hypothesis: Combine weekly trend bias with daily mean-reversion zones. In strong weekly uptrend, buy dips to daily S1/S2; in strong weekly downtrend, sell rallies to daily R1/R2. Use volume to filter false signals. Works in bull (trend continuation) and bear (mean reversion at extremes) markets.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_breakout_with_reversion_zone_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for reversion zones
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLC for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Standard pivot point (using previous day)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_open = np.roll(open_1d, 1)
    # Handle first values
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    prev_open[0] = open_1d[0]
    
    # Pivot point and support/resistance levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    
    # Align daily levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Weekly trend: 21-period EMA
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25  # Maintain position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R1 or weekly trend turns down
            if close[i] >= r1_aligned[i] or close[i] < ema_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
                
        elif position == -1:  # Short position
            # Exit: price reaches S1 or weekly trend turns up
            if close[i] <= s1_aligned[i] or close[i] > ema_21_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: weekly uptrend + price at S1/S2 with volume
            if (close[i] > ema_21_aligned[i] and 
                (close[i] <= s1_aligned[i] * 1.005 or close[i] <= s2_aligned[i] * 1.01) and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: weekly downtrend + price at R1/R2 with volume
            elif (close[i] < ema_21_aligned[i] and 
                  (close[i] >= r1_aligned[i] * 0.995 or close[i] >= r2_aligned[i] * 0.99) and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals