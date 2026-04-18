#!/usr/bin/env python3
"""
12h_1W_1D_Camarilla_R1S1_Breakout_Volume_V1
Hypothesis: Use weekly and daily confluence for directional bias with 12H entry, requiring price to break above weekly R1 AND daily R1 (or below S1) with volume > 2.0x average to reduce trade frequency. This dual-timeframe filter ensures strong trends and avoids false breakouts in sideways markets.
Long when price breaks above both weekly R1 and daily R1 with volume confirmation.
Short when price breaks below both weekly S1 and daily S1 with volume confirmation.
Fixed position size 0.25. Uses volume confirmation only (no additional filters) to keep trade count low.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via dual timeframe confluence and volume confirmation.
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
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe bias
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for intermediate timeframe bias
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly OHLC for Camarilla calculation
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Daily OHLC for Camarilla calculation
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous week's OHLC for weekly Camarilla
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    # Previous day's OHLC for daily Camarilla
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Weekly Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1w = prev_high_1w - prev_low_1w
    r1_1w = prev_close_1w + range_1w * 1.1 / 12
    s1_1w = prev_close_1w - range_1w * 1.1 / 12
    
    # Daily Camarilla levels
    range_1d = prev_high_1d - prev_low_1d
    r1_1d = prev_close_1d + range_1d * 1.1 / 12
    s1_1d = prev_close_1d - range_1d * 1.1 / 12
    
    # Align weekly and daily data to 12h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above BOTH weekly R1 and daily R1 with volume
            if (close[i] > r1_1w_aligned[i] and close[i] > r1_1d_aligned[i] and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below BOTH weekly S1 and daily S1 with volume
            elif (close[i] < s1_1w_aligned[i] and close[i] < s1_1d_aligned[i] and vol_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below EITHER weekly R1 or daily R1
            if close[i] < r1_1w_aligned[i] or close[i] < r1_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above EITHER weekly S1 or daily S1
            if close[i] > s1_1w_aligned[i] or close[i] > s1_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1W_1D_Camarilla_R1S1_Breakout_Volume_V1"
timeframe = "12h"
leverage = 1.0