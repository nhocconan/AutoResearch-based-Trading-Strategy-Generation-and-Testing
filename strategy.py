#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
Designed for 1d timeframe targeting 30-100 total trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn.
In trending regimes (price > weekly EMA50 for longs, < for shorts),
Donchian breakouts with volume spike capture strong momentum continuations.
Exit on trend reversal (price crosses weekly EMA50) or opposite Donchian breakout.
Works in both bull and bear markets by following the weekly trend.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need weekly EMA50, Donchian(20), volume avg
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        upper_val = upper[i]
        lower_val = lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with EMA alignment and volume spike
            long_condition = (close_val > upper_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < lower_val and 
                             close_val < ema_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA50 (trend reversal)
            # or opposite Donchian breakout (lower band)
            if close_val < ema_val or close_val < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA50 (trend reversal)
            # or opposite Donchian breakout (upper band)
            if close_val > ema_val or close_val > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0