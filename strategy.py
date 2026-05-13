#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). Designed for BTC/ETH robustness: Camarilla levels provide institutional reference points, EMA50 ensures trend alignment, volume filter confirms participation. Targets 20-50 trades/year on 4h timeframe with discrete position sizing (0.25) to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeConfirm_v1"
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
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Calculate 1d Camarilla levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above Camarilla R1, price > 12h EMA50, volume spike (>1.5x avg)
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S1, price < 12h EMA50, volume spike (>1.5x avg)
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla S1 (mean reversion target) OR breaks below 12h EMA50
            if (close[i] < camarilla_s1_aligned[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla R1 (mean reversion target) OR breaks above 12h EMA50
            if (close[i] > camarilla_r1_aligned[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals