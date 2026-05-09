#!/usr/bin/env python3
# Hypothesis: 4h timeframe with 1-day Williams Alligator and 4-hour momentum confirmation.
# In trending markets, Alligator lines (jaws, teeth, lips) align in order; in ranging markets, they intertwine.
# Enters long when price > Alligator teeth and 4h ROC > 0, short when price < Alligator teeth and 4h ROC < 0.
# Uses 1-day ADX > 25 as trend filter to avoid false signals in low volatility.
# Exits when price crosses Alligator teeth or ADX drops below 20.
# Target: 80-150 total trades over 4 years (20-37/year) with size 0.25.

name = "4h_Alligator_ROC_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Williams Alligator (13,8,5 SMAs with future shift)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    # Jaws: 13-period SMA, shifted 8 bars forward
    jaws = close_1d.rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = close_1d.rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = close_1d.rolling(window=5, min_periods=5).mean().shift(3)
    
    jaws_values = jaws.values
    teeth_values = teeth.values
    lips_values = lips.values
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_values)
    
    # Calculate 1-day ADX (14-period) for trend strength
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up = high_1d - high_1d.shift(1)
    down = low_1d.shift(1) - low_1d
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 4-hour Rate of Change (10-period) for momentum
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    close_4h = df_4h['close']
    roc = close_4h.pct_change(periods=10) * 100
    roc_values = roc.values
    roc_aligned = align_htf_to_ltf(prices, df_4h, roc_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(teeth_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(roc_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Alligator teeth, ADX > 25 (trending), ROC > 0 (bullish momentum)
            if close[i] > teeth_aligned[i] and adx_aligned[i] > 25 and roc_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Alligator teeth, ADX > 25 (trending), ROC < 0 (bearish momentum)
            elif close[i] < teeth_aligned[i] and adx_aligned[i] > 25 and roc_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Alligator teeth OR ADX drops below 20 (trend weakening)
            if close[i] < teeth_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Alligator teeth OR ADX drops below 20 (trend weakening)
            if close[i] > teeth_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals