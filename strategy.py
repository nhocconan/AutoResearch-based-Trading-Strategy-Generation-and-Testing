# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla levels: H3, L3 from previous day
    # Use 1d high/low/close to calculate today's Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day
    camarilla_H3 = []
    camarilla_L3 = []
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_H3.append(np.nan)
            camarilla_L3.append(np.nan)
        else:
            # Previous day's range
            range_prev = high_1d[i-1] - low_1d[i-1]
            close_prev = close_1d[i-1]
            H3 = close_prev + range_prev * 1.1 / 4
            L3 = close_prev - range_prev * 1.1 / 4
            camarilla_H3.append(H3)
            camarilla_L3.append(L3)
    
    camarilla_H3 = np.array(camarilla_H3)
    camarilla_L3 = np.array(camarilla_L3)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_H3_12h = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_12h = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # 1w trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_H3_12h[i]) or np.isnan(camarilla_L3_12h[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 + uptrend + volume spike
            long_cond = (close[i] > camarilla_H3_12h[i]) and \
                        (close[i] > ema_50_1w_aligned[i]) and \
                        volume_spike[i]
            # Short: break below L3 + downtrend + volume spike
            short_cond = (close[i] < camarilla_L3_12h[i]) and \
                         (close[i] < ema_50_1w_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below L3 (mean reversion to mean)
            if close[i] < camarilla_L3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above H3
            if close[i] > camarilla_H3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals