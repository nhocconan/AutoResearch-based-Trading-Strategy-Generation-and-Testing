#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime
Hypothesis: 12h breakout at daily Camarilla R1/S1 levels with 1d trend filter (price > EMA34 for long, < EMA34 for short),
volume spike (>2x 20-period average) and chop regime filter (CHOP > 61.8 for ranging, < 38.2 for trending).
Designed for low trade frequency (<30/year) to minimize fee drag in BTC/ETH.
Works in both bull and bear markets by following the daily trend direction and using regime filter to avoid whipsaws.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime"
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
    
    # Get daily data for Camarilla calculation and EMA
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
    
    # Calculate Camarilla levels (R1, S1 are the key breakout levels)
    rang = ph - pl
    r1 = pc + 1.1 * rang * 1.0833  # R1 = Close + 1.1 * (High-Low) * 1.0833
    s1 = pc - 1.1 * rang * 1.0833  # S1 = Close - 1.1 * (High-Low) * 1.0833
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (ema_34_1d[i-1] * 33 + close_1d[i]) / 34
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Chop regime filter: calculate on 12h data
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    atr = np.full_like(close, np.nan)
    tr = np.full_like(close, np.nan)
    if len(close) >= 2:
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    if len(tr) >= 14:
        atr_sum = np.full_like(close, np.nan)
        atr_sum[13] = np.sum(tr[0:14])
        for i in range(14, len(close)):
            atr_sum[i] = atr_sum[i-1] - tr[i-14] + tr[i]
        
        max_high = np.full_like(close, np.nan)
        min_low = np.full_like(close, np.nan)
        if len(high) >= 14:
            max_high[13] = np.max(high[0:14])
            min_low[13] = np.min(low[0:14])
            for i in range(14, len(close)):
                max_high[i] = max(max_high[i-1], high[i])
                min_low[i] = min(min_low[i-1], low[i])
        
        chop = np.full_like(close, np.nan)
        valid_chop = (~np.isnan(atr_sum)) & (~np.isnan(max_high)) & (~np.isnan(min_low)) & ((max_high - min_low) > 0)
        chop[valid_chop] = 100 * np.log10(atr_sum[valid_chop] / (max_high[valid_chop] - min_low[valid_chop])) / np.log10(14)
    else:
        chop = np.full_like(close, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Ensure volume MA, EMA and CHOP are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ratio[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 AND uptrend (price > EMA34) AND volume spike AND trending regime (CHOP < 38.2)
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_ratio[i] > 2.0 and
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 AND downtrend (price < EMA34) AND volume spike AND trending regime (CHOP < 38.2)
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_ratio[i] > 2.0 and
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 OR trend reversal (price < EMA34) OR ranging regime (CHOP > 61.8)
            if close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 OR trend reversal (price > EMA34) OR ranging regime (CHOP > 61.8)
            if close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals