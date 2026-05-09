#!/usr/bin/env python3
# 12h_Camarilla_R2_S2_Breakout_1wEMA50_Trend_Volume
# Hypothesis: 12h timeframe with Camarilla R2/S2 breakout, weekly EMA50 trend filter, and volume spike confirmation.
# Uses weekly EMA50 for stronger trend filter (less whipsaw in chop) and monthly volatility for volume threshold.
# Weekly trend filter avoids counter-trend trades, volume confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Camarilla_R2_S2_Breakout_1wEMA50_Trend_Volume"
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
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])  # previous high
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])   # previous low
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]]) # previous close
    
    # Calculate Camarilla levels (R2, S2 are the key breakout levels)
    rang = ph - pl
    r2 = pc + 1.1 * rang * 1.0833  # R2 = Close + 1.1 * (High-Low) * 1.0833
    s2 = pc - 1.1 * rang * 1.0833  # S2 = Close - 1.1 * (High-Low) * 1.0833
    
    # Align Camarilla levels to 12h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume spike filter: current volume / 50-period average volume (more stable)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 50:
        vol_ma[49] = np.mean(volume[0:50])
        for i in range(50, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 49 + volume[i]) / 50
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Ensure volume MA and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2 AND weekly uptrend (price > weekly EMA50) AND volume spike
            if (close[i] > r2_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 2.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 AND weekly downtrend (price < weekly EMA50) AND volume spike
            elif (close[i] < s2_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 2.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S2 OR trend reversal (price < weekly EMA50)
            if close[i] < s2_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R2 OR trend reversal (price > weekly EMA50)
            if close[i] > r2_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals