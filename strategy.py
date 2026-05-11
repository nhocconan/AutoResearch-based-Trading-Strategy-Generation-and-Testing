#!/usr/bin/env python3
# 12h_1d_Camarilla_R1_S1_Breakout_Volume
# Hypothesis: Uses daily Camarilla pivot levels (R1/S1) as support/resistance.
# On 12h timeframe, breakouts above R1 or below S1 are taken with volume confirmation.
# Targets breakouts with institutional volume in both bull and bear markets.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Target: 12-37 trades/year to minimize fee drag while capturing meaningful moves.

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla pivot levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Calculate R1 and S1 levels
    r1 = pivot + 1.1 * range_1d / 4
    s1 = pivot - 1.1 * range_1d / 4
    
    # Shift by 1 to use only completed daily candle (avoid look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Align daily R1/S1 to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # --- Volume confirmation (2x 30-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 30-period volume MA
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge
            if (close[i] > r1_aligned[i] and volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge
            elif (close[i] < s1_aligned[i] and volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below S1
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals