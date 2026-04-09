#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Hypothesis: Daily Camarilla pivot levels with weekly trend filter and volume confirmation.
# Long when price breaks above H3 resistance with price > weekly EMA50 and volume > 1.5x average.
# Short when price breaks below L3 support with price < weekly EMA50 and volume > 1.5x average.
# Camarilla levels provide strong intraday support/resistance; weekly EMA ensures alignment with higher timeframe trend.
# Volume confirmation reduces false breakouts. Works in both bull and bear markets by following the weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema = close_1w[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_1w[49] = ema
        for i in range(50, len(close_1w)):
            ema = (close_1w[i] - ema) * multiplier + ema
            ema_50_1w[i] = ema
    
    # Align weekly EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla pivot levels
    # Pivot point and support/resistance levels
    camarilla_pivot = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 1:  # Need previous day's data
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Calculate pivot point
            pivot = (prev_high + prev_low + prev_close) / 3
            camarilla_pivot[i] = pivot
            
            # Calculate Camarilla levels
            range_val = prev_high - prev_low
            camarilla_h3[i] = pivot + (range_val * 1.1 / 2)
            camarilla_l3[i] = pivot - (range_val * 1.1 / 2)
    
    # Volume confirmation: 20-day average
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
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L3 level
            if close[i] <= camarilla_l3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H3 level
            if close[i] >= camarilla_h3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with trend and volume filters
            if (close[i] > camarilla_h3[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with trend and volume filters
            elif (close[i] < camarilla_l3[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals