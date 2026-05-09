#!/usr/bin/env python3
# 1d_Camarilla_R2_S2_Breakout_1wEMA34_VolumeSpike
# Hypothesis: Daily Camarilla R2/S2 breakout with weekly EMA34 trend filter and volume spike confirmation.
# Uses weekly trend to avoid counter-trend trades, works in both bull and bear markets.
# Targets 15-25 trades/year to minimize fee drag while capturing significant moves.

name = "1d_Camarilla_R2_S2_Breakout_1wEMA34_VolumeSpike"
timeframe = "1d"
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
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (ema_34_1w[i-1] * 33 + close_1w[i]) / 34
    
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
    
    # Align Camarilla levels to daily timeframe (already aligned, but keep for consistency)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
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
    
    start_idx = max(20, 34)  # Ensure volume MA and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2 AND weekly uptrend (close > weekly EMA34) AND volume spike
            if (close[i] > r2_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 AND weekly downtrend (close < weekly EMA34) AND volume spike
            elif (close[i] < s2_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S2 OR weekly trend reversal (close < weekly EMA34)
            if close[i] < s2_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R2 OR weekly trend reversal (close > weekly EMA34)
            if close[i] > r2_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals