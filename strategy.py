#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot breakouts (R1/S1) combined with 12h EMA50 trend filter and volume confirmation provide institutional-grade entries. Works in bull markets by catching breakouts with momentum and in bear markets by fading false breakouts at key levels with volume divergence.
"""

name = "4H_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla levels (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_r2 = np.zeros_like(close_1d)
    camarilla_s2 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's values
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        range_val = ph - pl
        camarilla_r1[i] = pc + (range_val * 1.1 / 12)
        camarilla_s1[i] = pc - (range_val * 1.1 / 12)
        camarilla_r2[i] = pc + (range_val * 1.1 / 6)
        camarilla_s2[i] = pc - (range_val * 1.1 / 6)
    
    # Align Camarilla levels to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period volume average
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and 12h uptrend
            if (close[i] > r1_4h[i] and vol_spike and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and 12h downtrend
            elif (close[i] < s1_4h[i] and vol_spike and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or loss of trend/volume
            if (close[i] < s1_4h[i] or 
                close[i] < ema_50_12h_aligned[i] or
                not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or loss of trend/volume
            if (close[i] > r1_4h[i] or 
                close[i] > ema_50_12h_aligned[i] or
                not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals