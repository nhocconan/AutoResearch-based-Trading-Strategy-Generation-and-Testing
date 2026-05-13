#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike (>2.0x 20-bar avg volume).
# Uses Camarilla pivot levels from daily timeframe for precise entry/exit, 12h EMA50 for trend alignment,
# and volume confirmation to reduce false signals. Designed for low trade frequency (target 75-200 total over 4 years)
# to minimize fee drag while capturing strong momentum moves in both bull and bear markets via trend-following logic.

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_VolumeConfirm_v1"
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
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior 1d bar (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla R1 = close + (high - low) * 1.12
    # Camarilla S1 = close - (high - low) * 1.12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate average volume for confirmation (20-period LTF)
    lookback = 20
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1, close > 12h EMA50, volume spike (>2.0x avg)
            if (high[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1, close < 12h EMA50, volume spike (>2.0x avg)
            elif (low[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla S1 or volume drops
            if (low[i] < camarilla_s1_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla R1 or volume drops
            if (high[i] > camarilla_r1_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals