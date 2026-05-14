#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from 4h timeframe for entry/exit, combined with
1d EMA34 trend filter to avoid counter-trend trades. Volume spike confirms breakout momentum.
Designed for 4h timeframe to balance trade frequency and capture meaningful moves.
Target: 20-50 trades/year per symbol (75-200 total over 4 years).
Uses discrete position sizing (0.30) to balance return and fee drag.
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
    
    # Calculate 4h Camarilla pivot levels (R1, S1, R2, S2)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Camarilla levels for 4h timeframe
    range_4h = high_4h - low_4h
    camarilla_r1 = close_4h + (range_4h * 1.1 / 12)
    camarilla_s1 = close_4h - (range_4h * 1.1 / 12)
    camarilla_r2 = close_4h + (range_4h * 1.1 / 6)
    camarilla_s2 = close_4h - (range_4h * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe (previous 4h bar values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s2)
    
    # Calculate 1d EMA34 for primary trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend on 1d AND volume spike
            if close[i] > camarilla_r1_aligned[i] and trend_up and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Break below Camarilla S1 AND downtrend on 1d AND volume spike
            elif close[i] < camarilla_s1_aligned[i] and trend_down and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S1 for longs, R1 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S1
                if close[i] < camarilla_s1_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R1
                if close[i] > camarilla_r1_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0