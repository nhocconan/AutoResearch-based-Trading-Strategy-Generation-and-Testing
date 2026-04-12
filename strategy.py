#!/usr/bin/env python3
"""
1d_1w_Weekly_Pivot_Bounce
Hypothesis: On 1d timeframe, price reacts to weekly pivot levels (PP, R1, S1, R2, S2).
In ranging markets (Choppiness Index > 61.8), price tends to revert from R1/S1.
In trending markets (Choppiness Index < 38.2), price breaks R2/S2 and continues.
Uses weekly pivot points as support/resistance, Choppiness Index for regime filter,
and volume confirmation to avoid false breaks. Designed to work in both bull and bear
markets by adapting to regime. Target: 15-25 trades per year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Weekly_Pivot_Bounce"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, window=14):
    """Calculate Choppiness Index: higher = ranging, lower = trending"""
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Wilder's ATR smoothing
    atr[window-1] = np.mean(tr[1:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
    
    # True range sum over window
    tr_sum = np.zeros_like(close)
    for i in range(window-1, len(tr)):
        tr_sum[i] = np.sum(tr[i-window+1:i+1])
    
    # Max high - min low over window
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(window-1, len(high)):
        max_high[i] = np.max(high[i-window+1:i+1])
        min_low[i] = np.min(low[i-window+1:i+1])
    
    # Choppiness Index formula
    chop = np.zeros_like(close)
    for i in range(window-1, len(close)):
        if max_high[i] - min_low[i] != 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(window)
        else:
            chop[i] = 50  # neutral when no range
    return chop

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: PP, R1, S1, R2, S2"""
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    return pp, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D INDICATORS: Volume MA(20) for confirmation ===
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    
    # === 1W INDICATOR: Weekly pivot points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp, r1, s1, r2, s2 = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # === 1D INDICATOR: Choppiness Index for regime filter ===
    chop = calculate_choppiness(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(chop[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime classification
        ranging = chop[i] > 61.8  # choppy/ranging market
        trending = chop[i] < 38.2  # trending market
        
        # Price levels
        price = close[i]
        
        # Long signals
        long_signal = False
        if ranging and vol_confirm:
            # In ranging market: bounce off S1
            if price <= s1_aligned[i] * 1.002 and price >= s1_aligned[i] * 0.998:
                long_signal = True
        elif trending and vol_confirm:
            # In trending market: break above R2 with momentum
            if price > r2_aligned[i] * 1.005 and close[i] > close[i-1]:
                long_signal = True
        
        # Short signals
        short_signal = False
        if ranging and vol_confirm:
            # In ranging market: bounce off R1
            if price >= r1_aligned[i] * 0.998 and price <= r1_aligned[i] * 1.002:
                short_signal = True
        elif trending and vol_confirm:
            # In trending market: break below S2 with momentum
            if price < s2_aligned[i] * 0.995 and close[i] < close[i-1]:
                short_signal = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:  # long position
            # Exit when price reaches R1 (take profit) or breaks below S2 (stop)
            if price >= r1_aligned[i] * 0.995 or price < s2_aligned[i] * 0.995:
                exit_long = True
        elif position == -1:  # short position
            # Exit when price reaches S1 (take profit) or breaks above R2 (stop)
            if price <= s1_aligned[i] * 1.005 or price > r2_aligned[i] * 1.005:
                exit_short = True
        
        # Update position and signals
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals