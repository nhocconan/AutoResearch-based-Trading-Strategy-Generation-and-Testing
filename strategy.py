#!/usr/bin/env python3
# 1h_1d_Range_Breakout_Volume
# Hypothesis: In 1h timeframe, use 1d range (high-low) to detect breakouts with volume confirmation.
# Long when price breaks above 1d high + volume surge, short when breaks below 1d low + volume surge.
# Works in ranging markets (captures breakouts) and trending markets (follows momentum).
# Uses 1d range as volatility filter to avoid false breakouts in low volatility.
# Target: 15-30 trades/year by requiring both price breakout and volume surge.

name = "1h_1d_Range_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d range (high-low) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    range_1d = high_1d - low_1d
    
    # --- 1d high and low levels ---
    high_1d_level = high_1d
    low_1d_level = low_1d
    
    # Align 1d levels to 1h timeframe (wait for 1d bar to close)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d_level)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d_level)
    
    # --- Volume confirmation (volume > 24-period average) ---
    vol_ma = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_surge = volume > vol_ma * 1.5  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1d data (need at least 1 day)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(high_1d_aligned[i]) or
            np.isnan(low_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d high + volume surge
            if close[i] > high_1d_aligned[i] and vol_surge[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 1d low + volume surge
            elif close[i] < low_1d_aligned[i] and vol_surge[i]:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: price returns below 1d high OR volatility drops
                if close[i] < high_1d_aligned[i] or not vol_surge[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns above 1d low OR volatility drops
                if close[i] > low_1d_aligned[i] or not vol_surge[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals