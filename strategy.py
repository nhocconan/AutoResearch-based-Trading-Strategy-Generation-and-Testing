#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and 1w Trend Filter
Hypothesis: Donchian(20) breakouts on 1d capture momentum in trending markets. 
Weekly trend filter (EMA34) ensures trades align with higher timeframe momentum, 
while volume confirmation filters weak breakouts. Works in bull markets via upward 
breaks and in bear markets via downward breaks. Low trade frequency due to 
strict weekly trend alignment and volume filter.
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_1w_aligned[i]
        vol_ok = vol_confirm[i]
        upper_channel = high_roll[i]
        lower_channel = low_roll[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian with volume + weekly uptrend
            if vol_ok and close[i] > upper_channel and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian with volume + weekly downtrend
            elif vol_ok and close[i] < lower_channel and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below lower Donchian or weekly trend turns down
            if close[i] < lower_channel or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above upper Donchian or weekly trend turns up
            if close[i] > upper_channel or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0