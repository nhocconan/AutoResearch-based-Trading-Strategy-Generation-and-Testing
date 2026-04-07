#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend Filter and Volume Confirmation.
Long when price breaks above 4h Donchian Upper (20) with 1d uptrend and volume confirmation.
Short when price breaks below 4h Donchian Lower (20) with 1d downtrend and volume confirmation.
Exit when price crosses back below Donchian Middle (10) for long or above for short.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
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
    
    # === 1D EMA TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)
    
    # === 4H DONCHIAN CHANNELS (20-period) ===
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # === VOLUME CONFIRMATION (4H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(one_d_ema_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > one_d_ema_aligned[i]
        downtrend = close[i] < one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian Middle OR trend turns down
            if close[i] < donch_mid[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian Middle OR trend turns up
            if close[i] > donch_mid[i] or uptrend:
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
            if close[i] > donch_high[i] and uptrend:
                # Breakout above Donchian High in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and downtrend:
                # Breakdown below Donchian Low in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals