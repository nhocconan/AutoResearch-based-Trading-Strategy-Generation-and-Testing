#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Volume_Trend_v1
Hypothesis: On 4h timeframe, buy when price breaks above Donchian upper band (20-period) with volume confirmation and daily uptrend (close > EMA50), sell when price breaks below lower band with volume confirmation and daily downtrend (close < EMA50). Exit on opposite band touch. Uses 1d EMA50 for trend filter to avoid counter-trend trades. Designed for 2-4 trades/week (~100-200/year) to balance opportunity and fee drag.
"""

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
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(lookback - 1, len(high)):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = np.full_like(close_1d, np.nan)
    for i in range(49, len(close_1d)):
        ema50_1d[i] = np.mean(close_1d[i-49:i+1]) if i == 49 else ema50_1d[i-1] * 0.9607843137 + close_1d[i] * 0.0392156863
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period average volume on 4h
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, prices, vol_ma_20)  # same timeframe
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    start = max(lookback, 20, 50)  # Ensure all indicators are valid
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Long: break above upper band with volume confirmation and daily uptrend
            if (close[i] > highest_high[i] and volume_ratio > 1.5 and
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: break below lower band with volume confirmation and daily downtrend
            elif (close[i] < lowest_low[i] and volume_ratio > 1.5 and
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches or goes below lower band
            if close[i] <= lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches or goes above upper band
            if close[i] >= highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0