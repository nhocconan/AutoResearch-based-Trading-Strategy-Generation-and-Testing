#!/usr/bin/env python3
"""
4h_AdaptiveTrend_With_Volume_Regime_Filter
Hypothesis: Combines adaptive trend (KAMA) with volume confirmation and Choppiness Index regime filter to avoid whipsaws. Long when KAMA slope turns positive, volume > 1.5x average, and market is trending (CHOP < 38.2). Short when KAMA slope turns negative, volume > 1.5x average, and market is trending. Exit when KAMA slope reverses or regime shifts to choppy. Designed for 4h to capture strong trends while avoiding range-bound losses.
"""

name = "4h_AdaptiveTrend_With_Volume_Regime_Filter"
timeframe = "4h"
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
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate KAMA on 4h
    period = 10
    change = np.abs(np.subtract(close[period:], close[:-period]))
    
    # Calculate volatility (sum of absolute changes)
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros_like(close)
    for i in range(period, len(close)):
        price_change = np.abs(close[i] - close[i-period])
        sum_vol = volatility[i] - volatility[i-period]
        if sum_vol > 0:
            er[i] = price_change / sum_vol
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (direction)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Calculate Choppiness Index on 1d
    atr_period = 14
    tr1 = np.subtract(df_1d['high'], df_1d['low'])
    tr2 = np.abs(np.subtract(df_1d['high'], np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])))
    tr3 = np.abs(np.subtract(df_1d['low'], np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.zeros_like(df_1d['close'])
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    max_high = np.zeros_like(df_1d['high'])
    min_low = np.zeros_like(df_1d['low'])
    max_high[0] = df_1d['high'][0]
    min_low[0] = df_1d['low'][0]
    for i in range(1, len(df_1d)):
        max_high[i] = max(max_high[i-1], df_1d['high'][i])
        min_low[i] = min(min_low[i-1], df_1d['low'][i])
    
    chop = np.zeros_like(df_1d['close'])
    for i in range(atr_period, len(df_1d)):
        sum_atr = np.sum(atr[i-atr_period+1:i+1])
        highest_high = max_high[i]
        lowest_low = min_low[i]
        if highest_high > lowest_low:
            chop[i] = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(atr_period)
        else:
            chop[i] = 50  # neutral when no range
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20)  # Warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: trending when CHOP < 38.2
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: KAMA slope turns positive, volume confirmation, trending market
            if (kama_slope[i] > 0 and kama_slope[i-1] <= 0 and  # slope just turned positive
                vol_ratio[i] > 1.5 and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short: KAMA slope turns negative, volume confirmation, trending market
            elif (kama_slope[i] < 0 and kama_slope[i-1] >= 0 and  # slope just turned negative
                  vol_ratio[i] > 1.5 and 
                  is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA slope turns negative OR market becomes choppy
            if kama_slope[i] < 0 and kama_slope[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            elif not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA slope turns positive OR market becomes choppy
            if kama_slope[i] > 0 and kama_slope[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            elif not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals