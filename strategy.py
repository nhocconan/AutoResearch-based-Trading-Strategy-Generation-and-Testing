#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above R3 (1d) AND 12h EMA34 > 12h EMA50 (uptrend) AND 4h volume > 1.5x 20-bar average volume.
Short when price breaks below S3 (1d) AND 12h EMA34 < 12h EMA50 (downtrend) AND 4h volume > 1.5x 20-bar average volume.
Exit when price touches the 1d pivot point (PP) or opposite Camarilla level (S3 for long, R3 for short).
Uses 1d for Camarilla levels, 12h for trend filter, 4h for execution and volume confirmation.
Designed to capture institutional breakouts with trend alignment and volume confirmation. Target: 20-30 trades/year per symbol.
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R3/S3 are stronger breakout levels)
    # R3 = Close + 1.1*(High-Low)
    # S3 = Close - 1.1*(High-Low)
    # PP = (High + Low + Close)/3
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = ema34_12h > ema50_12h  # True for uptrend
    downtrend_12h = ema34_12h < ema50_12h  # True for downtrend
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    uptrend_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Exit conditions: touch pivot or opposite level
        touch_pp = abs(close[i] - pp_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < s3_aligned[i]) or \
                         (position == -1 and close[i] > r3_aligned[i])
        
        if position == 0:
            # Long: break above R3 with uptrend and volume confirmation
            if (breakout_r3 and uptrend_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend and volume confirmation
            elif (breakout_s3 and downtrend_aligned[i] and volume_confirmed):
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

name = "4h_Camarilla_R3S3_12hEMATrend_Volume"
timeframe = "4h"
leverage = 1.0