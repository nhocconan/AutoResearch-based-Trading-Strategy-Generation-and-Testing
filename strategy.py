#!/usr/bin/env python3
"""
4h_Breakout_TrendVolume_1dTrendFilter
Hypothesis: 4h price breaks above/below 20-bar Donchian channel with volume confirmation and 1d trend filter.
This structure captures trending moves while avoiding chop. Works in bull markets via breakouts and in bear markets via sharp reversals with volume.
Limits trades via strict entry conditions to control fee drag.
"""

name = "4h_Breakout_TrendVolume_1dTrendFilter"
timeframe = "4h"
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
    
    # 4h Donchian channel (20-period)
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest[i] = np.max(high[i-lookback+1:i+1])
        lowest[i] = np.min(low[i-lookback+1:i+1])
    
    # 4h EMA trend filter (34-period)
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 4h volume average (20-period) for spike detection
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema34[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout and volume conditions
        bullish_breakout = close[i] > highest[i-1]
        bearish_breakout = close[i] < lowest[i-1]
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: bullish breakout + volume spike + 1d uptrend
            if bullish_breakout and volume_spike and (ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # SHORT: bearish breakout + volume spike + 1d downtrend
            elif bearish_breakout and volume_spike and (ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish breakout or loss of volume/momentum
            if bearish_breakout or (close[i] < ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish breakout or loss of volume/momentum
            if bullish_breakout or (close[i] > ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals