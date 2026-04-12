#!/usr/bin/env python3
"""
6h_1d_keltner_breakout_with_volume_and_regime
Hypothesis: Keltner breakout with volume confirmation and market regime filter.
Keltner channels adapt to volatility, reducing false breakouts in low volatility.
Volume ensures conviction. Regime filter (ADX) avoids chop. Works in bull/bear by
adapting to volatility and using regime to filter noise.
Target: 20-50 trades/year (80-200 total over 4 years).
"""

name = "6h_1d_keltner_breakout_with_volume_and_regime"
timeframe = "6h"
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
    
    # Get daily data for Keltner and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Keltner Channel (20, 2.0)
    ma = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    atr = pd.Series(np.maximum(
        np.maximum(high_1d - low_1d,
                   np.abs(high_1d - np.roll(close_1d, 1))),
        np.abs(low_1d - np.roll(close_1d, 1))
    )).rolling(window=20, min_periods=20).mean().values
    
    upper = ma + 2.0 * atr
    lower = ma - 2.0 * atr
    
    # ADX for regime filter (14)
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    tr = np.maximum(
        np.maximum(high_1d - low_1d,
                   np.abs(high_1d - np.roll(close_1d, 1))),
        np.abs(low_1d - np.roll(close_1d, 1))
    )
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align to 6h
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when trending (ADX > 25)
        if adx_aligned[i] <= 25:
            # In chop, stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Long entry: close breaks above upper Keltner with volume
        if close[i] > upper_aligned[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below lower Keltner with volume
        elif close[i] < lower_aligned[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite side
        elif position == 1 and close[i] < ma_aligned[i]:  # Exit at middle line
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ma_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

# Align middle line for exit
    ma_aligned = align_htf_to_ltf(prices, df_1d, ma)