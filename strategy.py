#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_SR_Filter_v3
Hypothesis: Price breaking above Donchian(20) high or below low captures breakouts with momentum.
Volume spike confirms institutional participation. Only trade when price is above/below the 200-period
simple moving average to align with longer-term trend, reducing whipsaw in chop.
Designed for fewer trades (~20-40/year) with strong edge in both bull (breakouts) and bear (breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 x 20-period average volume
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma20)
    
    # 200-period SMA for trend filter
    close_series = pd.Series(close)
    sma200 = close_series.rolling(window=200, min_periods=200).mean().values
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 200)  # Donchian, volume MA, SMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(sma200[i]) or 
            np.isnan(sma50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + price > SMA200 + 1d uptrend
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and 
                close[i] > sma200[i] and 
                close[i] > sma50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + price < SMA200 + 1d downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and 
                  close[i] < sma200[i] and 
                  close[i] < sma50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_Volume_SR_Filter_v3"
timeframe = "4h"
leverage = 1.0