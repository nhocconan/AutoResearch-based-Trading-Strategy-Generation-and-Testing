#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA20_Trend_VolumeSpike
Hypothesis: Trade 1h Camarilla R1/S1 breakouts only when 4h EMA20 confirms trend (price above/below EMA) and volume spikes (>2.0x 20-bar MA). Camarilla levels from 1d provide institutional support/resistance. Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise. Discrete sizing 0.20 limits fee drag. Target 15-37 trades/year.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime64 arithmetic in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    camarilla_R1 = close_1d + camarilla_range
    camarilla_S1 = close_1d - camarilla_range
    
    # Align Camarilla levels to 1h (completed 1d bar only)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA20 on 4h
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA20 to 1h (completed 4h bar only)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (1d), EMA20 (4h), volume MA (20)
    start_idx = max(20, 20)  # 20 for EMA and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + above 4h EMA20 + volume spike
            long_setup = (close[i] > camarilla_R1_aligned[i]) and \
                         (close[i] > ema_20_4h_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Camarilla S1 + below 4h EMA20 + volume spike
            short_setup = (close[i] < camarilla_S1_aligned[i]) and \
                          (close[i] < ema_20_4h_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price closes below Camarilla S1 OR below 4h EMA20
            if (close[i] < camarilla_S1_aligned[i]) or \
               (close[i] < ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price closes above Camarilla R1 OR above 4h EMA20
            if (close[i] > camarilla_R1_aligned[i]) or \
               (close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA20_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0