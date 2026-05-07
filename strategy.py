#!/usr/bin/env python3
# 1d_Donchian_Breakout_Volume_Trend
# Hypothesis: Uses daily Donchian channel breakouts with weekly trend filter and volume confirmation.
# In bull markets, buy when price breaks above 20-day high with bullish weekly trend.
# In bear markets, sell when price breaks below 20-day low with bearish weekly trend.
# Volume confirmation reduces false breakouts. Target: 15-25 trades/year per symbol.

timeframe = "1d"
name = "1d_Donchian_Breakout_Volume_Trend"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 2x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high with volume, and weekly trend is bullish
            if (close[i] > high_20[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume, and weekly trend is bearish
            elif (close[i] < low_20[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below 20-day low (mean reversion)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above 20-day high (mean reversion)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals