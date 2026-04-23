#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 Breakout with 1w EMA34 Trend Filter and Volume Spike
- Uses tight entry conditions (Camarilla R3/S3 breakout + 1w EMA34 trend + volume > 1.8x 30-period MA)
- Designed for 12h timeframe to balance trade frequency and noise reduction
- Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag
- Works in both bull and bear markets via trend filter (1w EMA34) and volume confirmation
- Focus on Camarilla R3/S3 levels (outer pivot levels) for stronger breakouts with follow-through
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = df_1d['close'].iloc[0]
    prev_high_1d[0] = df_1d['high'].iloc[0]
    prev_low_1d[0] = df_1d['low'].iloc[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close_1d - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 30)  # need EMA34_1w, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R3 (breakout resistance) AND price > 1w EMA34 (uptrend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.30
                position = 1
            # Short: Close < S3 (breakdown support) AND price < 1w EMA34 (downtrend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Close back inside previous day's Camarilla H-L range OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < S3 (breakdown of support) OR price < 1w EMA34
                if close[i] < s3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > R3 (breakout of resistance) OR price > 1w EMA34
                if close[i] > r3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0