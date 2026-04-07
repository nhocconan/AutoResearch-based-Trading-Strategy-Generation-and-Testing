#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w Trend Filter
Breakout long when price > Donchian(20) high and weekly close > weekly SMA50
Breakout short when price < Donchian(20) low and weekly close < weekly SMA50
Exit on opposite Donchian breakout or trend reversal
Designed to capture trends in both bull and bear markets with controlled risk
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Donchian Channel (20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    sma_50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(sma_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR weekly trend turns bearish
            if close[i] < donch_low[i] or weekly_close[-1] < sma_50[-1]:  # Use last known weekly values
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR weekly trend turns bullish
            if close[i] > donch_high[i] or weekly_close[-1] > sma_50[-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high AND weekly bullish
            if close[i] > donch_high[i] and weekly_close[-1] > sma_50[-1]:
                position = 1
                signals[i] = 0.30
            # Short: price breaks below Donchian low AND weekly bearish
            elif close[i] < donch_low[i] and weekly_close[-1] < sma_50[-1]:
                position = -1
                signals[i] = -0.30
    
    return signals