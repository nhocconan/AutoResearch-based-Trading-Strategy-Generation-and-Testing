#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h EMA Trend Filter and Volume Confirmation
Hypothesis: Donchian(20) breakouts on 4h timeframe capture momentum in both bull and bear markets.
Use 12h EMA34 as trend filter to avoid counter-trend trades, and volume spike (2x 20-period average) 
to confirm breakout strength. Designed for low trade frequency with clear breakout edge.
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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_34 = ema_34_12h_aligned[i]
        
        # Calculate Donchian channels (20-period)
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above 12h EMA34
            if price > donchian_high and volume_spike[i] and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and below 12h EMA34
            elif price < donchian_low and volume_spike[i] and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below Donchian low or below 12h EMA34
            if price < donchian_low or price < ema_34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above Donchian high or above 12h EMA34
            if price > donchian_high or price > ema_34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0