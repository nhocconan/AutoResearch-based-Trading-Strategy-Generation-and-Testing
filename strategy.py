#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: On 12h timeframe, enter long when price breaks above daily Camarilla H5 level with volume > 1.5x MA and weekly uptrend (price > weekly EMA20), short when price breaks below daily L5 level with volume > 1.5x MA and weekly downtrend (price < weekly EMA20). Uses daily Camarilla levels from prior day and weekly EMA20 for trend. Volume filter ensures institutional participation. Weekly trend filter reduces whipsaw. Target: 12-37 trades per year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY INDICATORS: Prior day OHLC for Camarilla levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_H5 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_L5 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 2
    
    # Align to 12h timeframe
    camarilla_H5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H5)
    camarilla_L5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L5)
    
    # === WEEKLY INDICATOR: EMA(20) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: volume > 1.5 * average volume of prior 20 periods
    vol_ma = np.zeros_like(volume)
    vol_ma[20] = np.mean(volume[0:20])
    for i in range(21, len(volume)):
        vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(camarilla_H5_aligned[i]) or np.isnan(camarilla_L5_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filters
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
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