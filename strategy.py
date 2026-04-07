#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Weekly Trend and Volume Confirmation
# Hypothesis: Breakouts of 12-hour Donchian channels (20-period) in the direction of
# weekly EMA(40) trend, confirmed by above-average volume, work in both bull and bear
# markets by capturing strong momentum moves. Weekly trend filter ensures we only
# trade in the direction of the higher timeframe trend, reducing whipsaws.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_donchian20_weekly_trend_volume_v1"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 40:
        return np.zeros(n)
    
    # Weekly EMA(40) for trend filter
    close_weekly = df_weekly['close'].values
    ema_40_weekly = pd.Series(close_weekly).ewm(span=40, adjust=False).mean().values
    ema_40_12h = align_htf_to_ltf(prices, df_weekly, ema_40_weekly)
    
    # 12h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 12h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_40_12h[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price touches or crosses below 12h EMA(50) or trend changes
            ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().values[i]
            if close[i] < ema_50 or close[i] < ema_40_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches or crosses above 12h EMA(50) or trend changes
            ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().values[i]
            if close[i] > ema_50 or close[i] > ema_40_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of weekly trend with volume confirmation
            if vol_ok:
                if close[i] > ema_40_12h[i]:  # Weekly uptrend
                    if high[i] > high_roll[i]:  # Break above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Weekly downtrend
                    if low[i] < low_roll[i]:  # Break below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals