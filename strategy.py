#!/usr/bin/env python3
"""
6H Donchian Breakout + Daily Trend + Volume Confirmation
Hypothesis: 6-hour Donchian channel breakouts aligned with daily EMA trend and volume spikes capture momentum moves in both bull and bear markets. Uses tight entry conditions (breakout + trend + volume) to limit trades to 12-30 per year per symbol, reducing fee drag while maintaining edge. Works in bull via breakout continuation, in bear via faded false breaks at band extremes with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_daily_trend_volume_v1"
timeframe = "6h"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Daily EMA(50) for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter (>1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] <= lowest_low[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] >= highest_high[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long: price breaks above Donchian high with uptrend and volume
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short: price breaks below Donchian low with downtrend and volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals