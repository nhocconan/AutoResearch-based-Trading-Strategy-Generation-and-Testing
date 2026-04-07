#!/usr/bin/env python3
# 4H DONCHIAN BREAKOUT WITH VOLUME CONFIRMATION AND 12H TREND FILTER
# Long when price breaks above Donchian upper (20-period high) with volume expansion AND 12h EMA trend up
# Short when price breaks below Donchian lower (20-period low) with volume expansion AND 12h EMA trend down
# Exit when price crosses back to Donchian middle (10-period mean)
# Uses Donchian channels for clear breakout levels, reducing whipsaw in ranging markets.
# Target: 20-50 trades/year per symbol to avoid fee drag, with strong risk-adjusted returns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    # Upper: 20-period high, Lower: 20-period low, Middle: 10-period mean of high/low
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle = (pd.Series(high).rolling(window=10, min_periods=10).mean().values + 
              pd.Series(low).rolling(window=10, min_periods=10).mean().values) / 2
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 12h trend filter (EMA 21) ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(middle[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.2:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 12h trend filter
            if close[i] > upper[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                # Breakout above upper channel with rising 12h EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lower[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                # Breakdown below lower channel with falling 12h EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals