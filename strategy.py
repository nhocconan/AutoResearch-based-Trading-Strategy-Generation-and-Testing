#!/usr/bin/env python3
"""
4h_Volume_Spike_Donchian_Breakout_V1
Strategy: 4h Donchian breakout with volume spike and 12h trend filter.
Long: Price breaks above 20-period Donchian high with volume spike in uptrend.
Short: Price breaks below 20-period Donchian low with volume spike in downtrend.
Uses 12h EMA34 for trend filter to avoid whipsaws.
Designed for 4h timeframe: ~20-50 trades/year per symbol (80-200 total over 4 years).
Works in bull/bear via trend filter and volume confirmation.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 20-period Donchian channels (4h)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_12h_aligned[i] > close_12h[0]  # Simplified: above first value as proxy
        # Better: use slope of EMA
        if i >= 35:
            ema_now = ema_34_12h_aligned[i]
            ema_prev = ema_34_12h_aligned[i-1]
            uptrend = ema_now > ema_prev
            downtrend = ema_now < ema_prev
        else:
            uptrend = False
            downtrend = False
        
        # Breakout conditions
        breakout_long = high[i] > high_20[i-1]  # Break above previous period's high
        breakout_short = low[i] < low_20[i-1]   # Break below previous period's low
        
        if position == 0:
            # Long: uptrend + breakout + volume spike
            if uptrend and breakout_long and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + breakout + volume spike
            elif downtrend and breakout_short and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or breakdown below Donchian low
            if not uptrend or low[i] < low_20[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or breakout above Donchian high
            if not downtrend or high[i] > high_20[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Volume_Spike_Donchian_Breakout_V1"
timeframe = "4h"
leverage = 1.0