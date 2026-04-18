#!/usr/bin/env python3
"""
4h 12h Donchian Breakout with Volume Confirmation and ATR Stop
Hypothesis: Price breaking above/below 4h Donchian channels with 12h EMA trend filter and volume spikes
captures strong momentum moves. Works in bull markets via upward breaks and bear markets via downward breaks.
Low trade frequency due to strict breakout conditions and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_12h_aligned[i]
        vol_ok = vol_confirm[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian with volume + uptrend
            if vol_ok and close[i] > upper_channel and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian with volume + downtrend
            elif vol_ok and close[i] < lower_channel and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below lower Donchian or trend turns down
            if close[i] < lower_channel or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above upper Donchian or trend turns up
            if close[i] > upper_channel or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0