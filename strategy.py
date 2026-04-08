#!/usr/bin/env python3
# 12h_donchian_20_1d_trend_volume_v1
# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Works in bull/bear: trend filter ensures trend alignment, Donchian breakout captures momentum.
# Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag.

name = "12h_donchian_20_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    # Calculate rolling max/min
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high[lookback-1:] = high_series.rolling(window=lookback, min_periods=lookback).max().values
    lowest_low[lookback-1:] = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d EMA trend filter (50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(lookback, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below lower Donchian or trend fails
            if close[i] < lowest_low[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above upper Donchian or trend fails
            if close[i] > highest_high[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian with uptrend and volume
            if close[i] > highest_high[i] and close[i] > ema_1d_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian with downtrend and volume
            elif close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals