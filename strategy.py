#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_trend
Hypothesis: 12-hour Camarilla breakout with volume confirmation and daily trend filter.
Uses daily EMA200 to filter trades in trending markets, reducing false breakouts in chop.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drift.
Works in bull/bear by only taking breakouts aligned with higher timeframe trend.
"""

name = "12h_1d_camarilla_breakout_volume_trend"
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
    
    # Get daily data for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
    
    # Daily EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align Camarilla levels and EMA200 to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above R4 with volume and above daily EMA200
        if (close[i] > r4_aligned[i] and vol_confirm[i] and 
            close[i] > ema200_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below S4 with volume and below daily EMA200
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and 
              close[i] < ema200_aligned[i] and position != -1):
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