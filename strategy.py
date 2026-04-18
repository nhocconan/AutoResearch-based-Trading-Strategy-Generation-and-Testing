#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeSpike_1dEMA34_Trend_v1
Hypothesis: Camarilla R1/S1 breakouts with volume confirmation and 1-day EMA34 trend filter capture strong moves while avoiding false breakouts in chop. Works in bull via upward breakouts above R1 and in bear via downward breakouts below S1. Target: ~25 trades/year to minimize fee drag.
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
    
    # Daily Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day
    R1 = np.zeros_like(close_1d)
    S1 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        # Use previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        R1[i] = prev_close + (range_val * 1.1 / 12)
        S1[i] = prev_close - (range_val * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1-day EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above daily EMA34
            if price > r1 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and below daily EMA34
            elif price < s1 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below R1 or below daily EMA34
            if price < r1 or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above S1 or above daily EMA34
            if price > s1 or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_1dEMA34_Trend_v1"
timeframe = "4h"
leverage = 1.0