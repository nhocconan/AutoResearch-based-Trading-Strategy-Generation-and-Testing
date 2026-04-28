#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla pivot breakouts at R3/S3 levels on 6h timeframe with 12h EMA50 trend filter and volume spike work in both bull and bear markets. The 12h trend filter reduces whipsaw from counter-trend breakouts, volume confirms institutional participation, and R3/S3 levels provide institutional-grade support/resistance. Target: 15-30 trades/year per symbol to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point and ranges
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R3, R4, S3, S4
    r3_12h = pivot_12h + range_12h * 1.1000 / 4.0
    r4_12h = pivot_12h + range_12h * 1.1000 / 2.0
    s3_12h = pivot_12h - range_12h * 1.1000 / 4.0
    s4_12h = pivot_12h - range_12h * 1.1000 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_long = close[i] > r3_12h_aligned[i-1]  # Break above R3
        breakout_short = close[i] < s3_12h_aligned[i-1]  # Break below S3
        
        # Strong breakout confirmation (beyond R4/S4 for institutional interest)
        strong_breakout_long = close[i] > r4_12h_aligned[i-1]
        strong_breakout_short = close[i] < s4_12h_aligned[i-1]
        
        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions with volume confirmation and trend alignment
        long_entry = breakout_long and volume_spike[i] and uptrend
        short_entry = breakout_short and volume_spike[i] and downtrend
        
        # Strong breakout entries (higher conviction)
        strong_long_entry = strong_breakout_long and volume_spike[i] and uptrend
        strong_short_entry = strong_breakout_short and volume_spike[i] and downtrend
        
        # Exit on opposite breakout (reverse position)
        long_exit = breakout_short and volume_spike[i]
        short_exit = breakout_long and volume_spike[i]
        
        if (long_entry or strong_long_entry) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_entry or strong_short_entry) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0