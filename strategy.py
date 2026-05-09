#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wEMA20_Trend_VolumeS
# Hypothesis: Breakouts at weekly Camarilla R1/S1 levels with 1w EMA20 trend filter and volume spike confirmation on 12h timeframe.
# Weekly Camarilla levels provide stronger support/resistance than daily levels.
# EMA20 trend filter avoids counter-trend trades. Volume spike (>2x 20-period average) confirms breakout strength.
# Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull and bear markets.

name = "12h_Camarilla_R1_S1_Breakout_1wEMA20_Trend_VolumeS"
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
    
    # Get weekly data for Camarilla calculation and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for Camarilla calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Calculate Camarilla levels (R1, S1 are the key breakout levels)
    rang = ph - pl
    r1 = pc + 1.1 * rang * 1.0833  # R1 = Close + 1.1 * (High-Low) * 1.0833
    s1 = pc - 1.1 * rang * 1.0833  # S1 = Close - 1.1 * (High-Low) * 1.0833
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (ema_20_1w[i-1] * 19 + close_1w[i]) / 20
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 AND uptrend (price > EMA20) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 AND downtrend (price < EMA20) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 OR trend reversal (price < EMA20)
            if close[i] < s1_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 OR trend reversal (price > EMA20)
            if close[i] > r1_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals