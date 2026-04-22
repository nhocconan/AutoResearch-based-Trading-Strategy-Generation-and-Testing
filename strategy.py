#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla Pivot Breakout with 12-hour EMA50 trend filter and volume spike.
Long when price breaks above R1 with 12h EMA50 uptrend and volume > 1.5x 20-period average.
Short when price breaks below S1 with 12h EMA50 downtrend and volume > 1.5x 20-period average.
Exit when price returns to Pivot point or 12h EMA50 trend reverses.
Designed for low trade frequency (20-40 trades/year) by requiring multiple confirmations:
trend alignment, pivot level breakout, and volume spike. Works in both bull and bear markets
by following the 12h EMA50 trend direction, which adapts to market conditions.
"""

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
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 1d data for Camarilla pivot levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align 12h EMA50 to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Align 1d Camarilla levels to 4h (no extra delay needed as they're based on closed daily bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: 12h EMA50 uptrend + price breaks above R1 + volume spike
            if ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and close[i] > r1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: 12h EMA50 downtrend + price breaks below S1 + volume spike
            elif ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and close[i] < s1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to pivot or 12h EMA50 trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot or EMA50 turns down
                if close[i] <= pivot_aligned[i] or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot or EMA50 turns up
                if close[i] >= pivot_aligned[i] or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0