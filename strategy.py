#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian channel breakout with weekly trend filter.
Long when price breaks above Donchian(20) high and weekly close > weekly SMA50.
Short when price breaks below Donchian(20) low and weekly close < weekly SMA50.
Exit when price returns to Donchian(20) midpoint or weekly trend reverses.
Designed for low trade frequency by requiring both price breakout and weekly trend alignment.
Works in both bull and bear markets by following weekly trend while using daily breakout for entry.
"""

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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Daily Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_20
    donchian_low = low_20
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for weekly SMA50
        # Skip if weekly data not ready
        if np.isnan(sma50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high and weekly close > weekly SMA50
            if close[i] > donchian_high[i] and close_1w[i // 7] > sma50_1w[i // 7]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low and weekly close < weekly SMA50
            elif close[i] < donchian_low[i] and close_1w[i // 7] < sma50_1w[i // 7]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian midpoint OR weekly trend turns bearish
                if close[i] <= donchian_mid[i] or close_1w[i // 7] < sma50_1w[i // 7]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian midpoint OR weekly trend turns bullish
                if close[i] >= donchian_mid[i] or close_1w[i // 7] > sma50_1w[i // 7]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wSMA50_Trend"
timeframe = "1d"
leverage = 1.0