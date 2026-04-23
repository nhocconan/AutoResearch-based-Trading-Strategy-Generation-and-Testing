#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 Breakout with 4h EMA50 Trend Filter and Volume Spike
- Uses tight entry conditions (Camarilla R1/S1 breakout + 4h EMA50 trend + volume > 1.5x 20-period MA)
- Designed for 1h timeframe with 4h/1d HTF for signal direction to reduce trade frequency
- Session filter (08-20 UTC) to avoid low-liquidity hours
- Target: 15-37 trades/year per symbol (60-150 total over 4 years) to avoid fee drag
- Works in both bull and bear markets via trend filter (4h EMA50) and volume confirmation
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d data for Camarilla levels (previous day)
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
    r1 = prev_close_1d + camarilla_range * 1.1 / 12
    s1 = prev_close_1d - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50_4h, vol MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 (breakout resistance) AND price > 4h EMA50 (uptrend) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: Close < S1 (breakdown support) AND price < 4h EMA50 (downtrend) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Close back inside previous day's Camarilla H-L range OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < S1 (breakdown of support) OR price < 4h EMA50
                if close[i] < s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > R1 (breakout of resistance) OR price > 4h EMA50
                if close[i] > r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0