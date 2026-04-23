#!/usr/bin/env python3
"""
Hypothesis: 6h ADX + Williams Alligator confluence with 1d Elder Power regime filter.
Long when: ADX > 25 (trending), Alligator bullish (jaw < teeth < lips), and 1d Bull Power > 0.
Short when: ADX > 25 (trending), Alligator bearish (jaw > teeth > lips), and 1d Bear Power < 0.
Exit when ADX < 20 (trend weak) or Alligator reverses.
Uses 1d HTF for Elder Power to ensure alignment with higher timeframe momentum. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 6h ADX (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h Williams Alligator (jaw=13, teeth=8, lips=5, all shifted)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d Elder Power (Bull Power = high - EMA13, Bear Power = low - EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align 1d Elder Power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14+8, 13+8, 5+3, 13)  # ADX, Alligator, Elder Power
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions
        alligator_bullish = jaw[i] < teeth[i] < lips[i]
        alligator_bearish = jaw[i] > teeth[i] > lips[i]
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # 1d Elder Power regime
        bull_regime = bull_power_aligned[i] > 0
        bear_regime = bear_power_aligned[i] < 0
        
        if position == 0:
            # Long: Strong trend + Alligator bullish + Bull regime
            if strong_trend and alligator_bullish and bull_regime:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend + Alligator bearish + Bear regime
            elif strong_trend and alligator_bearish and bear_regime:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: weak trend OR Alligator reverses OR bull regime ends
                if weak_trend or not alligator_bullish or not bull_regime:
                    exit_signal = True
            elif position == -1:
                # Short exit: weak trend OR Alligator reverses OR bear regime ends
                if weak_trend or not alligator_bearish or not bear_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ADX_Alligator_ElderPower_Confluence"
timeframe = "6h"
leverage = 1.0