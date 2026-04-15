#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1-week Donchian Breakout with Volume Confirmation and ATR Filter
# Uses weekly high/low from previous week as support/resistance. Breakouts are traded
# only with volume > 1.5x median volume and ATR > 0.5 * ATR(50) to filter low volatility.
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for previous week's high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's high and low (shifted by 1 to avoid look-ahead)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    # Align previous week's high/low to 4h timeframe
    prev_high_1w_aligned = align_htf_to_ltf(prices, df_1w, prev_high_1w)
    prev_low_1w_aligned = align_htf_to_ltf(prices, df_1w, prev_low_1w)
    
    # Calculate ATR (14-period) on 4h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1w_aligned[i]) or np.isnan(prev_low_1w_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            continue
        
        # Volatility filter: avoid low volatility periods
        if atr[i] < 0.5 * atr_ma[i]:
            continue
        
        # Long entry: price breaks above previous week's high + volume confirmation
        if (close[i] > prev_high_1w_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous week's low + volume confirmation
        elif (close[i] < prev_low_1w_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout
        elif position == 1 and close[i] < prev_low_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > prev_high_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_1w_Donchian_Breakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0