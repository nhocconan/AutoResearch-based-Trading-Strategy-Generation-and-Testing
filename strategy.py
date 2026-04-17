#!/usr/bin/env python3
"""
6h_Williams_Alligator_ElderRay_v1
Williams Alligator (Jaw/Teeth/Lips) + Elder Ray (Bull/Bear Power) + EMA34 filter.
Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND close > EMA34.
Short when: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND close < EMA34.
Exit when: Alligator alignment breaks OR EMA34 cross fails.
Uses 1d timeframe for EMA34 trend filter.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === Williams Alligator (13,8,5) ===
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[8:]])  # shift 8 bars forward
    
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[5:]])  # shift 5 bars forward
    
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[3:]])  # shift 3 bars forward
    
    # === Elder Ray (Bull/Bear Power) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator alignment
        bullish_align = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_align = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish alignment + bull power > 0 + close > EMA34
            if bullish_align and (bull_power[i] > 0) and (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish alignment + bear power < 0 + close < EMA34
            elif bearish_align and (bear_power[i] < 0) and (close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: alignment breaks OR EMA34 cross fails
            if not bullish_align or (close[i] <= ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment breaks OR EMA34 cross fails
            if not bearish_align or (close[i] >= ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_ElderRay_v1"
timeframe = "6h"
leverage = 1.0