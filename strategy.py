#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + Trend Filter
Hypothesis: Donchian channel breakouts capture trend momentum. Volume spikes confirm institutional participation.
Trend filter (12h EMA) avoids counter-trend trades. Works in bull (breakouts up) and bear (breakouts down).
Target: 20-40 trades/year per symbol. Low frequency reduces fee drag.
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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_12h_aligned[i]
        up_breakout = close[i] > donchian_high[i-1]  # Break above prior high
        down_breakout = close[i] < donchian_low[i-1]  # Break below prior low
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long on upward breakout + volume spike + uptrend
            if up_breakout and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short on downward breakout + volume spike + downtrend
            elif down_breakout and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on downward breakout or trend change
            if down_breakout or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on upward breakout or trend change
            if up_breakout or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0