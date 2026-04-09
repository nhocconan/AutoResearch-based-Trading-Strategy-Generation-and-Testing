#!/usr/bin/env python3
# 6h_1d_donchian_breakout_v1
# Hypothesis: 6-hour Donchian(20) breakout with 1d trend filter and volume confirmation.
# Long when 1d trend is up (close > 1d EMA50) and price breaks above Donchian upper band with volume confirmation.
# Short when 1d trend is down (close < 1d EMA50) and price breaks below Donchian lower band with volume confirmation.
# Exit when price returns to the Donchian midline (average of upper and lower bands).
# Uses 1d trend for better trend alignment than weekly, reducing whipsaws in bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(n):
        if i >= donch_len - 1:
            start_idx = i - donch_len + 1
            upper[i] = np.max(high[start_idx:i+1])
            lower[i] = np.min(low[start_idx:i+1])
    
    midline = (upper + lower) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        alpha = 2 / (50 + 1)
        ema50_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 1d trend to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation - 20 period average
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
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midline[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below midline
            if close[i] <= midline[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above midline
            if close[i] >= midline[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: 1d trend up (close > 1d EMA50) AND price breaks above upper band with volume confirmation
            if close[i] > ema50_1d_aligned[i] and close[i] > upper[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: 1d trend down (close < 1d EMA50) AND price breaks below lower band with volume confirmation
            elif close[i] < ema50_1d_aligned[i] and close[i] < lower[i] and volume[i] > vol_ma_20[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals