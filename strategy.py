#!/usr/bin/env python3

"""
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend and volume confirmation.
This strategy captures momentum breakouts in the direction of the higher-timeframe trend.
Volume spikes confirm institutional interest. Weekly trend filter prevents counter-trend trades.
Designed to work in both bull and bear regimes by following the weekly trend.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

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
    
    # Load weekly data - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day volume average for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above 20-day high with bullish weekly trend and volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema34_1w_aligned[i] and  # Bullish trend: price above weekly EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with bearish weekly trend and volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema34_1w_aligned[i] and  # Bearish trend: price below weekly EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to the opposite Donchian level
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to or below 20-day low
                if close[i] <= low_min_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to or above 20-day high
                if close[i] >= high_max_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_20_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0