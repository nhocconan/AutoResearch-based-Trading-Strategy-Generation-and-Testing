#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w Trend Filter and Volume Confirmation.
Long when price breaks above 20-day high with 1w uptrend and volume > 1.5x average.
Short when price breaks below 20-day low with 1w downtrend and volume > 1.5x average.
Exit when price crosses opposite Donchian level (10-day) or trend reverses.
Target: 10-25 trades/year to stay under 100 total over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
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
    
    # === 1W TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_ema = pd.Series(one_w_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    one_w_ema_aligned = align_htf_to_ltf(prices, df_1w, one_w_ema)
    
    # === DAILY DONCHIAN CHANNELS ===
    # 20-day high/low for entry
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 10-day high/low for exit (opposite side)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(one_w_ema_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(high_10[i]) or np.isnan(low_10[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > one_w_ema_aligned[i]
        downtrend = close[i] < one_w_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 10-day low OR trend turns down
            if close[i] < low_10[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-day high OR trend turns up
            if close[i] > high_10[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (1.5x average)
            if volume[i] < 1.5 * vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with trend alignment
            if close[i] > high_20[i] and uptrend:
                # Breakout above 20-day high in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < low_20[i] and downtrend:
                # Breakdown below 20-day low in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals