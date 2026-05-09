#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Choppiness_Index_Donchian_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period)
    atr_period = 14
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    range_max_min = highest_high - lowest_low
    chop = 100 * np.log10(atr_safe * atr_period / range_max_min) / np.log10(atr_period)
    
    # Trend filter: 1w EMA100
    ema100_1w = pd.Series(df_1w['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align to daily
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # Donchian channels (20-period) for entry/exit
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, 20)  # Need enough data for EMA100 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or np.isnan(ema100_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        trend = ema100_1w_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Enter long in uptrend when choppy (mean reversion opportunity)
            if chop_val > 61.8 and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short in downtrend when choppy
            elif chop_val > 61.8 and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend turns down or chop ends
            if close[i] < trend or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend turns up or chop ends
            if close[i] > trend or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals