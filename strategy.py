#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_volume_v1
# Camarilla pivot levels from 1d provide strong support/resistance.
# Breakout above/below H4 or L4 with volume confirmation indicates institutional interest.
# Works in both bull and bear markets by capturing breakouts from key levels.
# Low trade frequency expected due to strict breakout + volume confirmation.
name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d: H4, L4, H3, L3
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    range_1d = high_1d - low_1d
    H4 = close_1d + range_1d * 1.1 / 2
    L4 = close_1d - range_1d * 1.1 / 2
    H3 = close_1d + range_1d * 1.1 / 4
    L3 = close_1d - range_1d * 1.1 / 4
    
    # Align 1d Camarilla levels to 4h timeframe (these levels are valid for the entire day)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any Camarilla level not ready
        if np.isnan(H4_4h[i]) or np.isnan(L4_4h[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume confirmation
        bullish_break = (close[i] > H4_4h[i]) and vol_confirm[i]
        bearish_break = (close[i] < L4_4h[i]) and vol_confirm[i]
        
        # Exit conditions: return to opposite H3/L3 level or opposite breakout
        exit_long = (close[i] < L3_4h[i]) or (bearish_break and vol_confirm[i])
        exit_short = (close[i] > H3_4h[i]) or (bullish_break and vol_confirm[i])
        
        if bullish_break and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_break and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals