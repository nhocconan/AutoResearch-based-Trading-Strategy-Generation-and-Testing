#!/usr/bin/env python3
"""
12h Donchian Breakout with 1w Trend Filter and Volume Confirmation
Hypothesis: 12h Donchian channel breakouts (20-period) capture medium-term trends.
1w EMA40 trend filter ensures trades align with weekly momentum.
Volume confirmation filters weak breakouts.
Works in bull markets via upward breaks and bear markets via downward breaks.
Designed for low trade frequency (~20-40/year) to minimize fee drag.
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
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        trend = ema40_1w_aligned[i]
        vol_ok = vol_confirm[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high with volume + uptrend
            if vol_ok and close[i] > upper and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low with volume + downtrend
            elif vol_ok and close[i] < lower and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Donchian low or trend turns down
            if close[i] < lower or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Donchian high or trend turns up
            if close[i] > upper or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0