#!/usr/bin/env python3
"""
12h_Donchian_20_TenkanKijun_Cross_24hTrend
Hypothesis: On 12h timeframe, use Donchian channel breakout (20-period) for entry, confirmed by Tenkan/Kijun cross (Ichimoku base/conversion line) and 24h EMA trend. Exit on opposite Donchian break. Designed for low trade frequency (~15-30/year) with trend alignment to work in both bull and bear markets. Uses 1d trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Donchian channel (20-period) on 12h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_period = 9
    tenkan_high = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    tenkan_low = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_period = 26
    kijun_high = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    kijun_low = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (kijun_high + kijun_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, kijun_period, 21)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(ema21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above Donchian high
        short_breakout = close[i] < lowest_low[i-1]   # Break below Donchian low
        
        # Tenkan/Kijun cross for momentum confirmation
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # 1d trend filter: EMA21 slope
        ema21_up = ema21_1d_aligned[i] > ema21_1d_aligned[i-1]
        ema21_down = ema21_1d_aligned[i] < ema21_1d_aligned[i-1]
        
        if long_breakout and tk_cross_up and ema21_up and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and tk_cross_down and ema21_down and position >= 0:
            signals[i] = -0.25
            position = -1
        elif close[i] < lowest_low[i-1] and position == 1:  # Exit long on Donchian low break
            signals[i] = -0.25
            position = 0
        elif close[i] > highest_high[i-1] and position == -1:  # Exit short on Donchian high break
            signals[i] = 0.25
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian_20_TenkanKijun_Cross_24hTrend"
timeframe = "12h"
leverage = 1.0