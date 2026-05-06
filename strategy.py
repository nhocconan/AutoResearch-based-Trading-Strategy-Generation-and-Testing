#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla R3/S3 pivot breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 level AND 1d EMA34 is rising AND 4h volume > 1.5 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 level AND 1d EMA34 is falling AND 4h volume > 1.5 * avg_volume(20)
# Exit when price returns to 1d Camarilla pivot point (mid)
# Uses discrete sizing 0.25 to balance performance and fee drag
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# Camarilla pivot provides intraday structure from higher timeframe (1d)
# 1d EMA34 ensures we trade with the daily trend while reducing whipsaws
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "4h_1dCamarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3, pivot point)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    R3_1d = close_1d + range_1d * 1.1 / 2  # Camarilla R3
    S3_1d = close_1d - range_1d * 1.1 / 2  # Camarilla S3
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3, EMA34 rising, volume spike
            if (close[i] > R3_1d_aligned[i] and close[i-1] <= R3_1d_aligned[i-1] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3, EMA34 falling, volume spike
            elif (close[i] < S3_1d_aligned[i] and close[i-1] >= S3_1d_aligned[i-1] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Camarilla pivot point
            if close[i] <= pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Camarilla pivot point
            if close[i] >= pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals