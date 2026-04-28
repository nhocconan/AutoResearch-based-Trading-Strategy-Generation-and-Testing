#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot R3/S3 breakout with volume spike and 1w EMA34 trend filter.
# Enter long when price breaks above Camarilla R3 with volume > 2.0x 20-bar average and price > 1w EMA34 (uptrend).
# Enter short when price breaks below Camarilla S3 with volume > 2.0x 20-bar average and price < 1w EMA34 (downtrend).
# Exit on opposite Camarilla level (S3 for long, R3 for short).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 50-150 total trades over 4 years.
# Camarilla levels provide structure from 1d, volume confirms breakout strength, 1w EMA34 filters counter-trend noise.
# Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > camarilla_r3_aligned[i] and volume_confirm[i] and close[i] > ema_34_1w_aligned[i]
        short_breakout = close[i] < camarilla_s3_aligned[i] and volume_confirm[i] and close[i] < ema_34_1w_aligned[i]
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_s3_aligned[i]
        short_exit = close[i] > camarilla_r3_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals