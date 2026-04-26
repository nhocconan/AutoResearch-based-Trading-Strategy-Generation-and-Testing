#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_HTFVolumeSpike_v1
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts aligned with 1-week trend (EMA34) and 1-week volume spike (volume > 2.0x 20-period EMA) capture strong multi-day trends while avoiding whipsaws. The 1-week HTF filter ensures we only trade in the direction of the primary trend, reducing false breakouts in choppy markets. Volume confirmation adds conviction. Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag and work in both bull and bear markets.
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
    
    # Load 1w data ONCE before loop for HTF trend filter (EMA34) and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1w EMA20 of volume for volume spike filter
    volume_1w = df_1w['volume'].values
    vol_ema_1w = pd.Series(volume_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ema_1w)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d close for Camarilla calculation
    close_1d_prev = np.concatenate([[np.nan], close_1d[:-1]])
    camarilla_range = high_1d - low_1d
    r1 = close_1d_prev + 1.1 * camarilla_range / 12
    s1 = close_1d_prev - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for EMA and Camarilla)
    start_idx = max(40, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ema_1w_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA34)
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # 1w volume confirmation: volume > 2.0x EMA20 of volume
        volume_spike = volume[i] > 2.0 * vol_ema_1w_aligned[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume
        if uptrend and volume_spike and breakout_r1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S1 in downtrend with volume
        elif downtrend and volume_spike and breakout_s1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_HTFVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0