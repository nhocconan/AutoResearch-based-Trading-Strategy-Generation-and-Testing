#!/usr/bin/env python3
# 6h_1d_volatility_breakout_v1
# Hypothesis: Breakouts from daily volatility bands (ATR-based) on 6h chart with trend filter from daily EMA50.
# Long when price breaks above upper band (close + ATR*multiplier) with price > daily EMA50.
# Short when price breaks below lower band (close - ATR*multiplier) with price < daily EMA50.
# Exit when price returns to the opposite band or daily close.
# Uses volatility expansion to capture momentum bursts in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
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
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR(14) for volatility bands
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(tr_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[:14])
        for i in range(14, len(tr_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate volatility bands: close ± ATR*multiplier
    multiplier = 2.0
    upper_band_1d = close_1d + atr_1d * multiplier
    lower_band_1d = close_1d - atr_1d * multiplier
    
    # Align bands to 6h timeframe
    upper_band_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_band_1d)
    lower_band_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_band_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_band_1d_aligned[i]) or 
            np.isnan(lower_band_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below lower band or daily close
            if (close[i] <= lower_band_1d_aligned[i] or 
                close[i] <= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above upper band or daily close
            if (close[i] >= upper_band_1d_aligned[i] or 
                close[i] >= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper band with trend filter
            if (close[i] > upper_band_1d_aligned[i] and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band with trend filter
            elif (close[i] < lower_band_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals