#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume spike (>2.0x 20-bar avg volume).
# Uses Camarilla pivot levels from weekly timeframe for precise entry/exit, 1w EMA50 for trend alignment,
# and volume confirmation to reduce false signals. Designed for low trade frequency (target 50-150 total over 4 years)
# to minimize fee drag while capturing strong momentum moves in both bull and bear markets via trend-following logic.

name = "12h_Camarilla_R1S1_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1w bar (HTF)
    df_1w_camarilla = get_htf_data(prices, '1w')
    if len(df_1w_camarilla) < 2:
        return np.zeros(n)
    high_1w = df_1w_camarilla['high'].values
    low_1w = df_1w_camarilla['low'].values
    close_1w_camarilla = df_1w_camarilla['close'].values
    # Camarilla R1 = close + (high - low) * 1.12
    # Camarilla S1 = close - (high - low) * 1.12
    camarilla_r1 = close_1w_camarilla + (high_1w - low_1w) * 1.12
    camarilla_s1 = close_1w_camarilla - (high_1w - low_1w) * 1.12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w_camarilla, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w_camarilla, camarilla_s1)
    
    # Calculate average volume for confirmation (20-period LTF)
    lookback = 20
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1, close > 1w EMA50, volume spike (>2.0x avg)
            if (high[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1, close < 1w EMA50, volume spike (>2.0x avg)
            elif (low[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
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