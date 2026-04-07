#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour trend filter, volume confirmation.
Long when price breaks above Donchian upper channel (20) in uptrend with volume confirmation.
Short when price breaks below Donchian lower channel (20) in downtrend with volume confirmation.
Exit on opposite Donchian breakout or trend reversal.
Uses 12h trend filter to reduce whipsaw and capture larger trends.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # === 12H TREND FILTER (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    twelve_h_close = df_12h['close'].values
    twelve_h_ema = pd.Series(twelve_h_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    twelve_h_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_ema)
    
    # === DONCHIAN CHANNEL (LTF) ===
    dc_length = 20
    dc_upper = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    dc_lower = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(twelve_h_ema_aligned[i]) or np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > twelve_h_ema_aligned[i]
        downtrend = close[i] < twelve_h_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend turns down
            if close[i] < dc_lower[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend turns up
            if close[i] > dc_upper[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with trend alignment
            if uptrend and close[i] > dc_upper[i]:
                position = 1
                signals[i] = 0.25
            elif downtrend and close[i] < dc_lower[i]:
                position = -1
                signals[i] = -0.25
    
    return signals