#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from 6h timeframe for entry/exit, combined with
1d EMA50 trend filter to avoid counter-trend trades. Volume spike confirms breakout momentum.
Designed for 6h timeframe to reduce trade frequency and fee drag while capturing meaningful moves.
Works in both bull and bear markets by following the 1d trend direction.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to balance return and fee drag.
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
    
    # Calculate 6h Camarilla pivot levels (R3, S3, R4, S4)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    open_6h = df_6h['open'].values
    
    # Camarilla levels for 6h timeframe
    range_6h = high_6h - low_6h
    camarilla_r3 = close_6h + (range_6h * 1.1 / 4)
    camarilla_s3 = close_6h - (range_6h * 1.1 / 4)
    camarilla_r4 = close_6h + (range_6h * 1.1 / 2)
    camarilla_s4 = close_6h - (range_6h * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (previous 6h bar values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s4)
    
    # Calculate 1d EMA50 for primary trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend on 1d AND volume spike
            if close[i] > camarilla_r3_aligned[i] and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND downtrend on 1d AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S3 for longs, R3 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Camarilla S3
                if close[i] < camarilla_s3_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Camarilla R3
                if close[i] > camarilla_r3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0