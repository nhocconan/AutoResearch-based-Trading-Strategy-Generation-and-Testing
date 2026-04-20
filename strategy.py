# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_Regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot Points (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot (same as classic)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === 4h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Chopiness Index (1d) for regime filter ===
    # Calculate daily chopiness index: higher = ranging, lower = trending
    high_low = df_1d['high'].values - df_1d['low'].values
    atr1 = np.zeros_like(high_low)
    atr1[0] = high_low[0]
    for i in range(1, len(high_low)):
        tr = max(
            high_low[i],
            abs(df_1d['high'].values[i] - df_1d['close'].values[i-1]),
            abs(df_1d['low'].values[i] - df_1d['close'].values[i-1])
        )
        atr1[i] = 0.9 * atr1[i-1] + 0.1 * tr
    
    # Sum of absolute returns over 14 days
    abs_returns = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_abs_ret = np.zeros_like(abs_returns)
    for i in range(len(abs_returns)):
        if i < 14:
            sum_abs_ret[i] = np.sum(abs_returns[:i+1])
        else:
            sum_abs_ret[i] = np.sum(abs_returns[i-13:i+1])
    
    # Chopiness index formula
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if sum_abs_ret[i] > 0 and atr1[i] > 0:
            chop[i] = 100 * np.log10(sum_abs_ret[i] / (atr1[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(80, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(pivot_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume confirmation in low chop (trending market)
            if close_val > r1_val and vol_ratio_val > 2.5 and chop_val < 40:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation in low chop
            elif close_val < s1_val and vol_ratio_val > 2.5 and chop_val < 40:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot OR chop increases (rangy market)
            if close_val < pivot_val or chop_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot OR chop increases
            if close_val > pivot_val or chop_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals