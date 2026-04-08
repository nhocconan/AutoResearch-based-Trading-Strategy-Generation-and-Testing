#!/usr/bin/env python3
"""
12h Donchian Breakout + Weekly EMA Trend + Volume Confirmation
Hypothesis: 12h Donchian(20) breakouts aligned with weekly EMA(21) trend and volume > 1.5x 20-period average capture strong momentum.
Designed for 12h timeframe to balance trade frequency (target 12-37/year) and signal quality in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_weekly_ema_volume_v1"
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
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend filter
    ema_21 = df_1w['close'].ewm(span=21, adjust=False).mean().values
    ema_21_12h = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume filter (>1.5x 20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
            
        # Calculate 12h Donchian(20) using only past data
        donchian_high = np.max(high[i-20:i]) if i >= 20 else np.nan
        donchian_low = np.min(low[i-20:i]) if i >= 20 else np.nan
        
        if np.isnan(donchian_high) or np.isnan(donchian_low):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA (trend reversal)
            if close[i] < ema_21_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA (trend reversal)
            if close[i] > ema_21_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at Donchian high with trend alignment
            if (close[i] >= donchian_high and 
                close[i] > ema_21_12h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at Donchian low with trend alignment
            elif (close[i] <= donchian_low and 
                  close[i] < ema_21_12h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals