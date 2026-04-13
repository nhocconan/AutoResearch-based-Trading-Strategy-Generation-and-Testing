#!/usr/bin/env python3
"""
12h_1d_Touch_Retest
Hypothesis: In 12h timeframe, price often retraces to test prior daily high/low levels after breaking them.
Enter long when price retraces to test broken daily high as support in uptrend (price > 50 EMA).
Enter short when price retraces to test broken daily low as resistance in downtrend (price < 50 EMA).
Use volume confirmation on retest and EMA trend filter to avoid counter-trend trades.
Target: 15-35 trades/year on 12h (60-140 total over 4 years).
Works in bull markets via long retests of daily highs, in bear markets via short retests of daily lows.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for reference levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 50 EMA on daily for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Track broken daily levels (levels that price has closed beyond)
    broken_high = np.full(len(close_1d), np.nan)  # broken daily highs acting as support
    broken_low = np.full(len(close_1d), np.nan)   # broken daily lows acting as resistance
    
    for i in range(1, len(close_1d)):
        # If daily close breaks above prior daily high, that high becomes potential support
        if close_1d[i] > high_1d[i-1]:
            broken_high[i] = high_1d[i-1]
        else:
            broken_high[i] = broken_high[i-1]
            
        # If daily close breaks below prior daily low, that low becomes potential resistance
        if close_1d[i] < low_1d[i-1]:
            broken_low[i] = low_1d[i-1]
        else:
            broken_low[i] = broken_low[i-1]
    
    # Align daily data to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    broken_high_aligned = align_htf_to_ltf(prices, df_1d, broken_high)
    broken_low_aligned = align_htf_to_ltf(prices, df_1d, broken_low)
    
    # Calculate 12h EMA for additional timing filter (optional)
    ema_20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(broken_high_aligned[i]) or np.isnan(broken_low_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = pd.Series(volume[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
            volume_expansion = volume[i] > (vol_ma_20 * 1.5)
        else:
            volume_expansion = False
        
        # Long setup: price > daily 50 EMA (uptrend) and retesting broken daily high
        if close[i] > ema_50_1d_aligned[i] and not np.isnan(broken_high_aligned[i]):
            # Price is near broken high level (within 0.5%)
            if abs(close[i] - broken_high_aligned[i]) / broken_high_aligned[i] < 0.005:
                if volume_expansion:
                    if position != 1:
                        position = 1
                        signals[i] = position_size
                    else:
                        signals[i] = position_size
                else:
                    # Hold position if already long
                    if position == 1:
                        signals[i] = position_size
                    else:
                        signals[i] = 0.0
            else:
                # Not at retest level - hold if long, otherwise flat
                if position == 1:
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
        
        # Short setup: price < daily 50 EMA (downtrend) and retesting broken daily low
        elif close[i] < ema_50_1d_aligned[i] and not np.isnan(broken_low_aligned[i]):
            # Price is near broken low level (within 0.5%)
            if abs(close[i] - broken_low_aligned[i]) / broken_low_aligned[i] < 0.005:
                if volume_expansion:
                    if position != -1:
                        position = -1
                        signals[i] = -position_size
                    else:
                        signals[i] = -position_size
                else:
                    # Hold position if already short
                    if position == -1:
                        signals[i] = -position_size
                    else:
                        signals[i] = 0.0
            else:
                # Not at retest level - hold if short, otherwise flat
                if position == -1:
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        
        # No clear setup - flatten
        else:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Touch_Retest"
timeframe = "12h"
leverage = 1.0