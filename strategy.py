#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike
Using 12h primary timeframe with Camarilla pivot levels (R1/S1) from prior day as 
key support/resistance. Breakout above R1 or below S1 with 1d EMA34 trend alignment 
and volume confirmation captures momentum moves. Discrete sizing 0.25 limits fee churn.
Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data)
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
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34_1d, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 (breakout resistance) AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 (breakdown support) AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside previous day's Camarilla H-L range OR loss of trend
            exit_signal = False
            if position == 1:
                # Exit long when close < S1 (breakdown of support) OR price < 1d EMA34
                if close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > R1 (breakout of resistance) OR price > 1d EMA34
                if close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0