#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: 12h Camarilla R1/S1 breakout with 1w trend filter (price > 1w EMA50) and volume confirmation.
Goes long when price breaks above R1 with bullish volume and 1w uptrend.
Short when price breaks below S1 with bearish volume and 1w downtrend.
Exit when price re-enters the Camarilla H-L range or on opposite breakout.
Uses discrete sizing (0.25) to minimize fees. Target: 12-37 trades/year.
Works in bull via breakouts with trend, in bear via mean reversion at extreme levels.
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
    
    # Get 12h data for Camarilla calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_width = 1.1 * (high_12h - low_12h) / 12.0
    r1_12h = close_12h + camarilla_width
    s1_12h = close_12h - camarilla_width
    h_l_range_12h = high_12h - low_12h  # For exit condition
    
    # Align Camarilla levels to original timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    h_l_range_12h_aligned = align_htf_to_ltf(prices, df_12h, h_l_range_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1, bullish volume, 1w uptrend
            long_signal = (close[i] > r1_12h_aligned[i]) and vol_spike[i] and (close[i] > ema_50_1w_aligned[i])
            # Short: price breaks below S1, bearish volume, 1w downtrend
            short_signal = (close[i] < s1_12h_aligned[i]) and vol_spike[i] and (close[i] < ema_50_1w_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price re-enters H-L range or breaks below S1 (contrarian exit)
            exit_signal = (close[i] <= high_12h_aligned[i] and close[i] >= low_12h_aligned[i]) or (close[i] < s1_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price re-enters H-L range or breaks above R1 (contrarian exit)
            exit_signal = (close[i] <= high_12h_aligned[i] and close[i] >= low_12h_aligned[i]) or (close[i] > r1_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0