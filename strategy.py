#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with volume confirmation and 1w trend filter.
Long when price breaks above R3 (1d) AND 1d volume > 1.5x 20-bar average AND 1w close > 1w EMA34.
Short when price breaks below S3 (1d) AND 1d volume > 1.5x 20-bar average AND 1w close < 1w EMA34.
Exit when price touches 1d pivot point (PP) or opposite Camarilla level (S3 for long, R3 for short).
Uses 1d for execution and Camarilla levels, 1w for trend filter. Target: 15-25 trades/year per symbol.
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
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels
    # R3 = Close + 1.1*(High-Low)/4
    # S3 = Close - 1.1*(High-Low)/4
    # PP = (High + Low + Close)/3
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng / 4
    s3 = close_1d - 1.1 * rng / 4
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 1d volume MA for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: 1w close above/below 1w EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Breakout conditions
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Exit conditions: touch pivot or opposite level
        touch_pp = abs(close[i] - pp_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < s3_aligned[i]) or \
                         (position == -1 and close[i] > r3_aligned[i])
        
        if position == 0:
            # Long: break above R3 with volume confirmation and uptrend
            if (breakout_r3 and volume_confirmed and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume confirmation and downtrend
            elif (breakout_s3 and volume_confirmed and downtrend):
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

name = "1d_Camarilla_R3S3_Volume_1wEMA34_Trend"
timeframe = "1d"
leverage = 1.0