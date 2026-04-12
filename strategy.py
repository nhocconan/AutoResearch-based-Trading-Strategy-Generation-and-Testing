#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_regime
Hypothesis: 12-hour Camarilla breakout with volume confirmation and choppy market filter.
Works in bull/bear by targeting breakouts in trending markets and avoiding false signals in chop.
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""

name = "12h_1d_camarilla_breakout_volume_regime"
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
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Choppy market filter: Chop index > 61.8 = range (avoid breakouts)
    def calculate_chop(high, low, close, window=14):
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean()
        max_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        min_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(window)
        return chop.fillna(50).values
    
    chop = calculate_chop(high_1d, low_1d, close_1d)
    chop_filter = chop < 61.8  # Only trade when NOT choppy (trending market)
    
    # Align Chop filter to 12h timeframe
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above R4 with volume and trend filter
        if (close[i] > r4_aligned[i] and vol_confirm[i] and 
            chop_filter_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below S4 with volume and trend filter
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and 
              chop_filter_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
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