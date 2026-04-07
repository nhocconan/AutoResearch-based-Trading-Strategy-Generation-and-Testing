#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
Breakouts above 20-day high (long) or below 20-day low (short) when price is above/below weekly EMA40,
with volume > 1.5x 20-day average. Uses discrete position sizing (0.25) to limit turnover.
Targets 7-25 trades/year (30-100 over 4 years). Works in bull markets via breakouts and bear
markets via short breakdowns, with volume and trend filters reducing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False).mean().values
    ema40_1d = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Daily Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema40_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR weekly trend turns down
            if close[i] < low_min[i] or close[i] < ema40_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR weekly trend turns up
            if close[i] > high_max[i] or close[i] > ema40_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long: price >= Donchian high, above weekly EMA, volume confirmation
            if (close[i] >= high_max[i] and 
                close[i] > ema40_1d[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Breakdown short: price <= Donchian low, below weekly EMA, volume confirmation
            elif (close[i] <= low_min[i] and 
                  close[i] < ema40_1d[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals