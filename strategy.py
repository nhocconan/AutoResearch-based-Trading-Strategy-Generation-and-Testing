#!/usr/bin/env python3
"""
12h_1w1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: 12h price breaks above/below weekly and daily combined resistance/support levels with 12h volume > 1.5x 20-period average.
Weekly/daily confluence acts as stronger support/resistance than single timeframe alone.
Long when price breaks above weekly R4 or daily R4 (whichever is higher) with volume confirmation.
Short when price breaks below weekly S4 or daily S4 (whichever is lower) with volume confirmation.
Exit when price returns to weekly/daily midpoint.
Designed for 12h timeframe to target 15-30 trades/year with strong trend capture in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    vol_1w = df_1w['volume'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    # Daily calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Weekly Camarilla
    range_1w = prev_high_1w - prev_low_1w
    camarilla_pp_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    camarilla_r4_1w = camarilla_pp_1w + (range_1w * 1.1 / 2)
    camarilla_s4_1w = camarilla_pp_1w - (range_1w * 1.1 / 2)
    
    # Daily Camarilla
    range_1d = prev_high_1d - prev_low_1d
    camarilla_pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_r4_1d = camarilla_pp_1d + (range_1d * 1.1 / 2)
    camarilla_s4_1d = camarilla_pp_1d - (range_1d * 1.1 / 2)
    
    # Align weekly data to 12h
    camarilla_pp_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp_1w)
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # Align daily data to 12h
    camarilla_pp_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp_1d)
    camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Volume data
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean()
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w.values)
    
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d.values)
    
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_pp_1w_aligned[i]) or np.isnan(camarilla_r4_1w_aligned[i]) or
            np.isnan(camarilla_s4_1w_aligned[i]) or np.isnan(camarilla_pp_1d_aligned[i]) or
            np.isnan(camarilla_r4_1d_aligned[i]) or np.isnan(camarilla_s4_1d_aligned[i]) or
            np.isnan(vol_ma_20_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Combined levels: weekly and daily confluence
        combined_r4 = max(camarilla_r4_1w_aligned[i], camarilla_r4_1d_aligned[i])
        combined_s4 = min(camarilla_s4_1w_aligned[i], camarilla_s4_1d_aligned[i])
        combined_pp = (camarilla_pp_1w_aligned[i] + camarilla_pp_1d_aligned[i]) / 2
        
        # Volume condition: either weekly or daily volume > 1.5x 20-period average
        vol_condition = (vol_1w_aligned[i] > vol_ma_20_1w_aligned[i] * 1.5) or \
                        (vol_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5)
        
        # Breakout conditions
        long_breakout = close[i] > combined_r4
        short_breakout = close[i] < combined_s4
        
        # Exit condition: return to midpoint
        long_exit = close[i] < combined_pp
        short_exit = close[i] > combined_pp
        
        if position == 0:
            if long_breakout and vol_condition:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0