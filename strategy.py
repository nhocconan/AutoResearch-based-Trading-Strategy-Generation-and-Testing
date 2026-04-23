#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 Trend Filter and Volume Spike
- Uses Williams Alligator (jaw=13, teeth=8, lips=5) on 12h for trend identification
- Entry: price outside Alligator mouth + aligned with 1d EMA50 trend + volume > 2x 20-period MA
- Exit: price re-enters Alligator mouth OR loss of 1d EMA50 trend
- Designed for 12h timeframe to capture medium-term trends with low trade frequency
- Williams Alligator is effective in both trending and ranging markets
- Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag
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
    
    # Calculate Williams Alligator on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Alligator lines: jaw (13), teeth (8), lips (5) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50_1d, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Alligator state: mouth open (trending) or closed (ranging)
        # Mouth open when lips, teeth, jaw are separated and ordered
        if position == 0:
            # Long: price above lips (uptrend) AND lips > teeth > jaw (aligned) AND price > 1d EMA50 AND volume spike
            if (close[i] > lips_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below lips (downtrend) AND lips < teeth < jaw (aligned) AND price < 1d EMA50 AND volume spike
            elif (close[i] < lips_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price re-enters Alligator mouth (between lips and jaw) OR loss of 1d EMA50 trend
            exit_signal = False
            if position == 1:
                # Exit long when price <= lips (re-enters mouth) OR price < 1d EMA50
                if close[i] <= lips_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price >= lips (re-enters mouth) OR price > 1d EMA50
                if close[i] >= lips_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0