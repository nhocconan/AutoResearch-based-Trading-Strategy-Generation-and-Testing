#!/usr/bin/env python3
# 12h_1D_Momentum_Breakout
# Strategy: Trade breakouts above 1d high with momentum confirmation
# Long when price breaks above 1d high and 12h RSI > 50
# Short when price breaks below 1d low and 12h RSI < 50
# Exit when price returns to 1d range or RSI crosses 50
# Uses 1d range breakout with 12h momentum filter to capture trends
# Designed for 12h timeframe with selective entries to minimize trade frequency

name = "12h_1D_Momentum_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d high and low for breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 1d high/low to 12h timeframe (available after 1d bar closes)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Calculate 12h RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[1:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    gain_smooth = wilders_smooth(gain, 14)
    loss_smooth = wilders_smooth(loss, 14)
    
    rs = np.where(loss_smooth != 0, gain_smooth / loss_smooth, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: break above 1d high with bullish momentum (RSI > 50)
            if high[i] > high_1d_aligned[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Enter short: break below 1d low with bearish momentum (RSI < 50)
            elif low[i] < low_1d_aligned[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 1d range or momentum turns bearish
            if low[i] < low_1d_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 1d range or momentum turns bullish
            if high[i] > high_1d_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals