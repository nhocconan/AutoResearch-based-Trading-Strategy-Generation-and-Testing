#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime
# Hypothesis: 12h breakout at daily Camarilla R1/S1 with daily trend filter and volume spike, using chop regime filter to avoid whipsaws.
# Designed for low trade frequency (<20/year) to minimize fee drag in BTC/ETH.
# Works in both bull and bear markets by following the daily trend and using chop filter for range markets.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and filters
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
    
    # Choppiness Index filter (using daily data)
    chop = np.full_like(close_1d, np.nan)
    if len(high_1d) >= 14 and len(low_1d) >= 14 and len(close_1d) >= 14:
        atr_14 = np.full_like(close_1d, np.nan)
        tr = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
        tr = np.concatenate([[np.nan], tr])
        for i in range(14, len(close_1d)):
            if i == 14:
                atr_14[i] = np.nanmean(tr[1:15])
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
        
        highest_high = np.full_like(high_1d, np.nan)
        lowest_low = np.full_like(low_1d, np.nan)
        for i in range(14, len(high_1d)):
            highest_high[i] = np.max(high_1d[i-13:i+1])
            lowest_low[i] = np.min(low_1d[i-13:i+1])
        
        chop_raw = 100 * np.log10((atr_14 * 14) / (highest_high - lowest_low)) / np.log10(14)
        chop[14:] = chop_raw[14:]
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ratio[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 AND uptrend (price > EMA34) AND volume spike AND chop < 61.8 (trending)
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_ratio[i] > 2.0 and
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 AND downtrend (price < EMA34) AND volume spike AND chop < 61.8 (trending)
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_ratio[i] > 2.0 and
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 OR trend reversal (price < EMA34) OR chop > 61.8 (ranging)
            if (close[i] < s1_aligned[i] or 
                close[i] < ema_34_1d_aligned[i] or
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 OR trend reversal (price > EMA34) OR chop > 61.8 (ranging)
            if (close[i] > r1_aligned[i] or 
                close[i] > ema_34_1d_aligned[i] or
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals