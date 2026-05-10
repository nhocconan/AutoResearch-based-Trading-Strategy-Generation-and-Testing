#!/usr/bin/env python3
# 12h_KAMA_Trend_With_Volume_Spike_and_CHOP_Filter
# Hypothesis: Kaufman's Adaptive Moving Average (KAMA) on 12h chart combined with volume confirmation and Choppiness Index regime filter.
# Uses KAMA's adaptive nature to capture trends in both bull and bear markets.
# Volume confirmation ensures participation, while Choppiness Index filters out range-bound periods.
# Designed for low trade frequency on 12h timeframe to minimize fee drag.

name = "12h_KAMA_Trend_With_Volume_Spike_and_CHOP_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA on 12h data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Handle the first 9 values where we don't have 10-period lookback
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start from the 10th element
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Get daily typical price and range for Choppiness Index
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    atr_1d = np.zeros(len(df_1d))
    tr = np.maximum(df_1d['high'] - df_1d['low'],
                    np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)),
                               np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr[0] = df_1d['high'][0] - df_1d['low'][0]  # first TR
    for i in range(1, len(df_1d)):
        atr_1d[i] = np.mean(tr[max(0, i-13):i+1])  # 14-period ATR
    
    # Calculate Choppiness Index (14-period)
    sum_tr_14 = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i >= 13:
            sum_tr_14[i] = np.sum(tr[i-13:i+1])
        else:
            sum_tr_14[i] = np.sum(tr[0:i+1])
    
    high_low_range_14 = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i >= 13:
            high_low_range_14[i] = np.max(df_1d['high'][i-13:i+1]) - np.min(df_1d['low'][i-13:i+1])
        else:
            high_low_range_14[i] = np.max(df_1d['high'][0:i+1]) - np.min(df_1d['low'][0:i+1])
    
    chop = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if high_low_range_14[i] != 0 and i >= 13:
            chop[i] = 100 * np.log10(sum_tr_14[i] / high_low_range_14[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral when undefined
    
    # Align KAMA and Choppiness Index to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)  # no extra delay needed
    
    # Volume confirmation (20-period MA on 12h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), volume MA (20), and Choppiness (need at least 14 days)
    start_idx = max(10, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price relative to KAMA
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Choppiness regime filter: only trade when trending (CHOP < 38.2)
        trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long entry: price above KAMA + volume spike + trending market
            if above_kama and volume_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + volume spike + trending market
            elif below_kama and volume_confirm and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or market becomes choppy
            if below_kama or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or market becomes choppy
            if above_kama or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals