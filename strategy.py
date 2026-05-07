#!/usr/bin/env python3
name = "12h_Donchian_20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels from previous 12h
    # Previous period's high, low from 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian(20): 20-period high and low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate upper and lower bands
    upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    # Volume spike: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Wait for Donchian and EMA34
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above upper band with volume spike in uptrend
            if close[i] > upper_aligned[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower band with volume spike in downtrend
            elif close[i] < lower_aligned[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below lower band or trend turns down
            if close[i] < lower_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above upper band or trend turns up
            if close[i] > upper_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on 12h with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above upper band (bullish breakout) with volume spike in 1d uptrend.
# Short when price breaks below lower band (bearish breakdown) with volume spike in 1d downtrend.
# Uses discrete position size (0.25) to minimize churn. Target 15-30 trades/year.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Volume spike (>1.8x average) ensures conviction behind the move.
# Designed for 12h timeframe to target 60-120 total trades over 4 years, avoiding overtrading.