#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and 1-Week Trend Filter
Hypothesis: Donchian breakouts capture breakout momentum. Volume confirmation ensures institutional participation.
The 1-week EMA filter ensures we only trade in the direction of the higher-timeframe trend, improving win rate in both bull and bear markets.
Designed for low trade frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week EMA40 for trend filter
    ema40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Donchian channel (20-period) on 12h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema40_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        trend = ema40_1w_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long on Donchian breakout above upper band + volume confirmation + uptrend
            if close[i] > donch_high and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakout below lower band + volume confirmation + downtrend
            elif close[i] < donch_low and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on Donchian breakdown below lower band
            if close[i] < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on Donchian breakout above upper band
            if close[i] > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_1wTrend"
timeframe = "12h"
leverage = 1.0
EOF