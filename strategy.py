#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h EMA crossover + volume spike + 1d ADX trend filter
# Uses EMA crossover for trend changes, volume to confirm momentum,
# and ADX to ensure trading only in trending markets. Works in bull/bear
# by taking crossovers only when ADX > 25. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for price action and EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Load 12h data for faster EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA9 on 4h and EMA21 on 12h for crossover
    ema9_4h = pd.Series(close_4h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / (tr14 + 1e-10)
    minus_di = 100 * minus_dm14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align indicators to 4h timeframe
    ema9_4h_aligned = align_htf_to_ltf(prices, df_4h, ema9_4h)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema9_4h_aligned[i]) or np.isnan(ema21_12h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: EMA9_4h crosses above EMA21_12h + volume spike + ADX > 25
        if (ema9_4h_aligned[i] > ema21_12h_aligned[i] and
            ema9_4h_aligned[i-1] <= ema21_12h_aligned[i-1] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: EMA9_4h crosses below EMA21_12h + volume spike + ADX > 25
        elif (ema9_4h_aligned[i] < ema21_12h_aligned[i] and
              ema9_4h_aligned[i-1] >= ema21_12h_aligned[i-1] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse crossover or ADX < 20 (trend weakening)
        elif position == 1 and (ema9_4h_aligned[i] < ema21_12h_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ema9_4h_aligned[i] > ema21_12h_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_12h_EMA_Crossover_Volume_ADX"
timeframe = "4h"
leverage = 1.0