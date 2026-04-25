#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirmation
Hypothesis: 12h Camarilla R1/S1 breakout confirmed by 1w EMA50 trend and volume spikes (>2.0x 20-bar avg). 
Enters long when price breaks above R1 with volume in 1w uptrend (price > EMA50), short when price breaks below S1 with volume in 1w downtrend (price < EMA50). 
Camarilla levels from daily pivot provide precise support/resistance; breakouts with volume and weekly trend capture momentum. 
Designed for 12h timeframe with ~15-30 trades/year, avoiding overtrading via strict confluence. 
Works in both bull/bear: trend filter adapts direction, volume confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Camarilla pivot levels (previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 based on previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1_level = close_1d + camarilla_range
    s1_level = close_1d - camarilla_range
    
    # Align Camarilla levels to 12h timeframe (1-day delay for previous day's data)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and Camarilla alignment
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and 1w uptrend
            long_setup = (close[i] > r1_aligned[i]) and volume_spike[i] and (close[i] > ema_50_1w_aligned[i])
            # Short: price breaks below S1 with volume spike and 1w downtrend
            short_setup = (close[i] < s1_aligned[i]) and volume_spike[i] and (close[i] < ema_50_1w_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend turns down
            if (close[i] < s1_aligned[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend turns up
            if (close[i] > r1_aligned[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0