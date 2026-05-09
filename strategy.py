#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Regime
Hypothesis: Trade breakouts at Camarilla R1/S1 levels on 12h timeframe with 1d EMA trend filter and volume spike confirmation.
Uses 1d Choppiness Index regime filter to avoid whipsaw in sideways markets. Designed for low trade frequency (12-37/year)
to minimize fee drag. Works in both bull and bear markets by following 1d trend direction and avoiding range-bound periods.
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
    
    # Get 1d data for trend filter, Choppiness Index, and Camarilla calculation
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
    
    # Calculate Choppiness Index (14-period)
    def calculate_choppiness(high, low, close, period=14):
        n = len(high)
        atr = np.full(n, np.nan)
        if n >= 1:
            tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
            tr = np.concatenate([[np.inf], tr])  # first TR undefined
            atr_period = np.full(n, np.nan)
            if n >= period:
                atr_period[period-1] = np.mean(tr[1:period+1])  # skip first undefined TR
                for i in range(period, n):
                    atr_period[i] = (atr_period[i-1] * (period-1) + tr[i]) / period
                atr = atr_period
        # Sum of true ranges over period
        sum_tr = np.full(n, np.nan)
        if n >= period:
            for i in range(period-1, n):
                start_idx = i - period + 1
                sum_tr[i] = np.sum(tr[start_idx:i+1])
        # Highest high and lowest low over period
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        if n >= period:
            for i in range(period-1, n):
                start_idx = i - period + 1
                highest_high[i] = np.max(high[start_idx:i+1])
                lowest_low[i] = np.min(low[start_idx:i+1])
        # Choppiness Index
        chop = np.full(n, np.nan)
        valid = (~np.isnan(sum_tr)) & (~np.isnan(highest_high)) & (~np.isnan(lowest_low)) & ((highest_high - lowest_low) > 0)
        chop[valid] = 100 * np.log10(sum_tr[valid] / ((highest_high[valid] - lowest_low[valid]) * period)) / np.log10(period)
        return chop
    
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (ema_34_1d[i-1] * 33 + close_1d[i]) / 34
    
    # Calculate Camarilla levels (R1, S1)
    rang = ph - pl
    r1 = pc + 1.1 * rang * 1.0833  # R1 = Close + 1.1 * (High-Low) * 1.0833
    s1 = pc - 1.1 * rang * 1.0833  # S1 = Close - 1.1 * (High-Low) * 1.0833
    
    # Align 1d indicators to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
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
    bars_since_entry = 0
    
    start_idx = max(34, 20)  # Ensure EMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Regime filter: only trade when market is trending (Choppiness Index < 38.2)
        is_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0 and is_trending:
            # Enter long: price breaks above R1 AND uptrend (price > EMA34) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below S1 AND downtrend (price < EMA34) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit long: price breaks below S1 OR trend reversal (price < EMA34) OR market becomes ranging
            if (chop_1d_aligned[i] >= 38.2 or  # regime change to ranging
                close[i] < s1_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 OR trend reversal (price > EMA34) OR market becomes ranging
            if (chop_1d_aligned[i] >= 38.2 or  # regime change to ranging
                close[i] > r1_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals