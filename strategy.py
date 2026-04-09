#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v22
# Hypothesis: 4-hour breakout of daily Camarilla levels with daily EMA50 trend filter and volume confirmation.
# Long when price breaks above H4 resistance with price > daily EMA50 and volume > 2.0x 20-bar average.
# Short when price breaks below L4 support with price < daily EMA50 and volume > 2.0x 20-bar average.
# Exit when price returns to opposite Camarilla level (L4 for longs, H4 for shorts).
# Position size fixed at 0.25 to limit drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme levels.
# Improved: Added stricter volume filter (2.0x) and trend filter confirmation to reduce trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v22"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema = np.mean(close_1d[:50])  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = ema
        for i in range(50, len(close_1d)):
            ema = (close_1d[i] - ema) * multiplier + ema
            ema_50_1d[i] = ema
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        c = df_1d['close'].iloc[i]
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        camarilla_h4[i] = c + 1.1 * (h - l) / 2
        camarilla_l4[i] = c - 1.1 * (h - l) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L4 level
            if close[i] <= camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H4 level
            if close[i] >= camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H4 with trend and volume filters
            if (close[i] > camarilla_h4_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > vol_ma_20[i] * 2.0):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L4 with trend and volume filters
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 2.0):
                position = -1
                signals[i] = -0.25
    
    return signals