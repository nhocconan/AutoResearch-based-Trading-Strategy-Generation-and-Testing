#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v1
Hypothesis: Donchian channel breakout on 4h with daily trend filter and volume confirmation.
Works in both bull and bear markets by following the trend defined by daily EMA.
Breakouts provide clear entry/exit signals with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4h Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            start_idx = i - lookback + 1
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian lower or trend turns bearish
            if close[i] <= lowest_low[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper or trend turns bullish
            if close[i] >= highest_high[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper with volume and bullish trend
            if (close[i] >= highest_high[i] and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower with volume and bearish trend
            elif (close[i] <= lowest_low[i] and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals