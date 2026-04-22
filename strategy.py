#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian breakout with 1-day trend filter and volume confirmation.
Donchian channels identify breakouts with clear support/resistance levels.
Daily trend filter avoids counter-trend trades. Volume spikes confirm institutional interest.
Designed for low trade frequency (15-25/year) to minimize fee drag in 12h timeframe.
Should work in both bull and bear regimes by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily trend using EMA crossover
    close_1d = df_1d['close'].values
    ema_fast = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slow = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).mean().values
    daily_trend = ema_fast > ema_slow  # True for uptrend, False for downtrend
    
    # Align daily trend to 12h timeframe
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(55, n):  # Start after slow EMA warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(daily_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high, daily uptrend, volume spike
            if (close[i] > donchian_high[i] and 
                daily_trend_aligned[i] > 0.5 and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low, daily downtrend, volume spike
            elif (close[i] < donchian_low[i] and 
                  daily_trend_aligned[i] < 0.5 and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: return to middle of Donchian channel or opposite breakout
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            
            if position == 1:
                # Exit long: price returns to midpoint or breaks below low
                if close[i] <= donchian_mid or close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to midpoint or breaks above high
                if close[i] >= donchian_mid or close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0