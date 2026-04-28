#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla pivot breakouts at R1/S1 levels on 4h timeframe with 1d EMA34 trend filter and volume spike work in both bull and bear markets. The 1d trend filter reduces whipsaw from counter-trend breakouts, volume confirms institutional participation, and R1/S1 levels provide institutional-grade support/resistance. Target: 20-50 trades/year per symbol to minimize fee drag while capturing significant moves.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, R2, S1, S2
    r1_1d = pivot_1d + range_1d * 1.1000 / 12.0
    r2_1d = pivot_1d + range_1d * 1.1000 / 6.0
    s1_1d = pivot_1d - range_1d * 1.1000 / 12.0
    s2_1d = pivot_1d - range_1d * 1.1000 / 6.0
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_long = close[i] > r1_1d_aligned[i-1]  # Break above R1
        breakout_short = close[i] < s1_1d_aligned[i-1]  # Break below S1
        
        # Strong breakout confirmation (beyond R2/S2 for institutional interest)
        strong_breakout_long = close[i] > r2_1d_aligned[i-1]
        strong_breakout_short = close[i] < s2_1d_aligned[i-1]
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0