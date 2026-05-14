#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 (1d) AND close > 1d EMA50 (uptrend) AND 6h volume > 2.0x 20-bar average volume.
Short when price breaks below S3 (1d) AND close < 1d EMA50 (downtrend) AND 6h volume > 2.0x 20-bar average volume.
Exit when price touches the 1d pivot point (PP) or opposite Camarilla level (S3 for long, R3 for short).
Uses 1d for Camarilla levels/EMA/trend, 6h for execution and volume confirmation.
Designed to capture strong breakouts in trending markets with volume confirmation. Target: 12-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    # R3 = Close + 1.1*(High-Low)
    # S3 = Close - 1.1*(High-Low)
    # PP = (High + Low + Close)/3
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 20-bar average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Exit conditions: touch pivot or opposite level
        touch_pp = abs(close[i] - pp_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < s3_aligned[i]) or \
                         (position == -1 and close[i] > r3_aligned[i])
        
        if position == 0:
            # Long: break above R3 with volume confirmation and uptrend (close > EMA50)
            if (breakout_r3 and volume_confirmed and close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume confirmation and downtrend (close < EMA50)
            elif (breakout_s3 and volume_confirmed and close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch pivot or break below S3
            if (touch_pp or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch pivot or break above R3
            if (touch_pp or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Volume_EMA50_Trend"
timeframe = "6h"
leverage = 1.0