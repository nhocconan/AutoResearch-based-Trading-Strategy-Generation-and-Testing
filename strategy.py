#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
- Long: Close breaks above Donchian(20) high + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period average
- Short: Close breaks below Donchian(20) low + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period average
- Exit: Opposite Donchian breakout or close crosses 1d EMA34
- Uses Donchian channels for structure, 1d EMA34 for higher-timeframe trend, volume spike for confirmation
- Discrete position sizing (0.25) to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
- Works in both bull and bear markets by trading with the 1d trend on breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # Donchian needs 20, EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close breaks above Donchian high + uptrend + volume spike
        long_signal = (close[i] > donchian_high[i-1] and  # Break above previous period's high
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        # Short: Close breaks below Donchian low + downtrend + volume spike
        short_signal = (close[i] < donchian_low[i-1] and  # Break below previous period's low
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Opposite Donchian breakout or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: Price breaks below Donchian low OR trend turns down
                if (close[i] < donchian_low[i-1] or 
                    close[i] < ema34_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Price breaks above Donchian high OR trend turns up
                if (close[i] > donchian_high[i-1] or 
                    close[i] > ema34_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0