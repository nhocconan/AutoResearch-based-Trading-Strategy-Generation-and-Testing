#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian(20) breakout + 1w EMA(100) trend filter + volume confirmation
# Hypothesis: In strong weekly trends, daily breakouts from 20-day channels have higher success.
# Works in both bull and bear by only taking breakouts in direction of weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_donchian20_1wema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 100:
        return np.zeros(n)
    
    # Weekly EMA(100) for trend filter
    close_weekly = df_weekly['close'].values
    ema_100_weekly = pd.Series(close_weekly).ewm(span=100, adjust=False).mean().values
    ema_100_daily = align_htf_to_ltf(prices, df_weekly, ema_100_weekly)
    
    # Daily Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: daily volume > 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_100_daily[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 20-day low or trend changes
            if close[i] < low_20[i] or close[i] < ema_100_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above 20-day high or trend changes
            if close[i] > high_20[i] or close[i] > ema_100_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of weekly trend with volume confirmation
            if vol_ok:
                if close[i] > ema_100_daily[i]:  # Uptrend
                    if high[i] > high_20[i]:  # Break above 20-day high
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] < low_20[i]:  # Break below 20-day low
                        position = -1
                        signals[i] = -0.25
    
    return signals