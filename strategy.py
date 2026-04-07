#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation.
Long when price breaks above Donchian(20) high with 1d uptrend and volume confirmation.
Short when price breaks below Donchian(20) low with 1d downtrend and volume confirmation.
Exit when price crosses the Donchian midpoint or trend reverses.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
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
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)
    
    # === DONCHIAN CHANNEL (12H) ===
    # Use 20-period lookback (equivalent to ~10 days on 12h)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # === VOLUME CONFIRMATION (12H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if (np.isnan(one_d_ema_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > one_d_ema_aligned[i]
        downtrend = close[i] < one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint OR trend turns down
            if close[i] < donchian_mid[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint OR trend turns up
            if close[i] > donchian_mid[i] or uptrend:
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
            if close[i] > highest_high[i] and uptrend:
                # Breakout above Donchian high in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and downtrend:
                # Breakdown below Donchian low in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals