#!/usr/bin/env python3
"""
4h Donchian Breakout with 12h Trend Filter and Volume Confirmation.
Long when price breaks above Donchian(20) upper band with 12h uptrend and volume confirmation.
Short when price breaks below Donchian(20) lower band with 12h downtrend and volume confirmation.
Exit when price crosses back below Donchian(20) middle line (long) or above (short).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # === DONCHIAN CHANNELS (4H) ===
    # Upper band: 20-period high
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle line: average of upper and lower
    donch_mid = (donch_high + donch_low) / 2
    
    # === VOLUME CONFIRMATION (4H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(twelve_h_ema_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > twelve_h_ema_aligned[i]
        downtrend = close[i] < twelve_h_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below middle line OR trend turns down
            if close[i] < donch_mid[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle line OR trend turns up
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
                # Breakout above upper band in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and downtrend:
                # Breakdown below lower band in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals