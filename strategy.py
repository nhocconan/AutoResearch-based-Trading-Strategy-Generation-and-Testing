#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_Filter_v1
# Hypothesis: KAMA adapts to market noise, providing a reliable trend signal in both trending and ranging markets.
# Combining KAMA trend direction with volume confirmation and a 4h Donchian(20) breakout filter creates high-probability entries.
# The strategy reduces whipsaws by requiring volume > 1.5x 20-period average and only taking trades in the direction of the KAMA trend.
# Designed for ~20-40 trades/year per symbol to minimize fee drag while capturing sustained moves.

name = "4h_KAMA_Trend_With_Volume_Filter_v1"
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
    
    # KAMA trend filter (using 4h data)
    def calculate_kama(close_series, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close_series, n=er_length))
        volatility = np.sum(np.abs(np.diff(close_series)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close_series)
        kama[0] = close_series[0]
        for i in range(1, len(close_series)):
            kama[i] = kama[i-1] + sc[i] * (close_series[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close)
    
    # 4h Donchian(20) for breakout filter
    def donchian_channels(high, low, length):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(len(high)):
            if i < length:
                upper[i] = np.max(high[:i+1])
                lower[i] = np.min(low[:i+1])
            else:
                upper[i] = np.max(high[i-length+1:i+1])
                lower[i] = np.min(low[i-length+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (er_length=10) and Donchian (20) and volume MA (20)
    start_idx = max(20, 20)  # Donchian and volume MA both need 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below KAMA
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        # Volume confirmation (>1.5x MA to filter low-volume noise)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above Donchian upper + volume
            if uptrend and close[i] > donchian_upper[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below Donchian lower + volume
            elif downtrend and close[i] < donchian_lower[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below Donchian upper
            if not uptrend or close[i] < donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above Donchian lower
            if not downtrend or close[i] > donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals