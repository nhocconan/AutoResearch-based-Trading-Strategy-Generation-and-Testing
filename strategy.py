#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Volume_v1
Hypothesis: On 4h timeframe, enter long when price breaks above 12h Camarilla H5 level with volume confirmation and price > 12h EMA20 (uptrend), enter short when price breaks below 12h L5 level with volume confirmation and price < 12h EMA20 (downtrend). Uses 12h EMA20 for trend filter and 12h Camarilla levels for structure. Volume filter ensures breakouts have institutional participation. Target: 20-35 trades per year per symbol (80-140 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Breakout_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H INDICATORS: EMA(20) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === 12H INDICATOR: Prior period OHLC for Camarilla levels ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each period
    camarilla_H5 = close_12h + 1.1 * (high_12h - low_12h) * 1.1 / 2
    camarilla_L5 = close_12h - 1.1 * (high_12h - low_12h) * 1.1 / 2
    
    # Align to 4h timeframe
    camarilla_H5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_H5)
    camarilla_L5_aligned = align_htf_to_ltf(prices, df_12h, camarilla_L5)
    
    # Volume filter: volume > 1.3 * average volume of prior 20 periods
    vol_ma = np.zeros_like(volume)
    vol_ma[20] = np.mean(volume[0:20])
    for i in range(21, len(volume)):
        vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if np.isnan(ema_20_12h_aligned[i]) or np.isnan(camarilla_H5_aligned[i]) or np.isnan(camarilla_L5_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filters
        uptrend = close[i] > ema_20_12h_aligned[i]
        downtrend = close[i] < ema_20_12h_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = (close[i] > camarilla_H5_aligned[i]) and volume_filter[i]
        short_breakout = (close[i] < camarilla_L5_aligned[i]) and volume_filter[i]
        
        # Exit conditions: trend reversal
        exit_long = not uptrend
        exit_short = not downtrend
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals