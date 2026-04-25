#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: 12h Camarilla R1/S1 breakout with 1-week EMA50 trend filter and volume confirmation.
Long when price breaks above R1 with 1w uptrend and volume spike. Short when price breaks below S1 with 1w downtrend and volume spike.
Using 12h timeframe reduces trade frequency to avoid fee drag. Weekly trend filter ensures trading with higher timeframe trend.
Volume confirmation reduces false breakouts. Discrete position sizing (0.25) minimizes churn.
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
    
    # Weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Camarilla calculation (based on prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current day using prior day's OHLC
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 0.275 * range_1d
    camarilla_s1 = close_1d - 0.275 * range_1d
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day for proper timing)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50(1w) and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 + 1w uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and volume_spike[i]
            # Short: break below S1 + 1w downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_50_1w_aligned[i]) and volume_spike[i]
            
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
            # Exit: price closes below R1 OR 1w trend turns down
            if (close[i] < camarilla_r1_aligned[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above S1 OR 1w trend turns up
            if (close[i] > camarilla_s1_aligned[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0