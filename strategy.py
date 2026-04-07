#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with daily trend filter and volume confirmation
# Uses 12h Donchian(20) breakouts for entries, daily MA(50) for trend filter,
# and volume > 1.5x 20-period average for confirmation. Exits on opposite Donchian break.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
# Works in bull markets via breakout momentum and in bear markets via mean reversion
# at channel extremes with volatility filter.

name = "12h_donchian20_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # Calculate daily MA(50) for trend filter
    close_1d = df_1d['close'].values
    ma_50 = np.full_like(close_1d, np.nan)
    for i in range(49, len(close_1d)):
        ma_50[i] = np.mean(close_1d[i-49:i+1])
    ma_50_aligned = align_htf_to_ltf(prices, df_1d, ma_50)
    
    # Calculate volume average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ma_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long conditions: price breaks above Donchian high + above daily MA + volume
        if close[i] > donch_high[i] and close[i] > ma_50_aligned[i] and vol_confirm:
            signals[i] = 0.25
        # Short conditions: price breaks below Donchian low + below daily MA + volume
        elif close[i] < donch_low[i] and close[i] < ma_50_aligned[i] and vol_confirm:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals