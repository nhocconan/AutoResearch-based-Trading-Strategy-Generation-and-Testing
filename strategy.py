#!/usr/bin/env python3
name = "12H_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA10 for trend filter
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align weekly EMA10 to 12h timeframe
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema10_1w_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 10-period average volume
        if i >= 10:
            avg_volume = np.mean(volume[i-10:i])
            volume_confirm = volume[i] > avg_volume * 1.5
        else:
            volume_confirm = False
        
        if position == 0:
            # Enter long: price breaks above R1 + weekly uptrend + volume confirmation
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema10_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + weekly downtrend + volume confirmation
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema10_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or weekly trend turns down
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or weekly trend turns up
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals