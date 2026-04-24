#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter and volume confirmation.
- Elder Ray Bull Power = high - EMA13(close); Bear Power = EMA13(close) - low
- Long when Bull Power > 0 AND Bear Power rising (less negative) AND ADX > 25 (trending) AND volume > 1.5x 20-period average
- Short when Bear Power < 0 AND Bull Power falling (less positive) AND ADX > 25 (trending) AND volume > 1.5x 20-period average
- Uses 6h primary timeframe with 12h HTF for ADX regime filter to target 50-150 trades over 4 years (12-37/year)
- Elder Ray measures price power relative to EMA, capturing trend strength
- ADX filter ensures we only trade in trending markets, avoiding chop
- Volume confirmation ensures legitimacy of price moves
"""

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
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: high - EMA
    bear_power = ema_13 - low   # Bear Power: EMA - low
    
    # Rate of change of Elder Ray components (to detect improving/weakening power)
    bull_power_roc = np.diff(bull_power, prepend=bull_power[0])
    bear_power_roc = np.diff(bear_power, prepend=bear_power[0])
    
    # Get 12h data ONCE before loop for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 14, 14) + 10  # Extra buffer for ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power_roc[i]) or np.isnan(bear_power_roc[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND improving (ROC > 0) AND Bear Power improving (ROC < 0, becoming less negative)
            # AND ADX > 25 (trending market) AND volume confirmation
            if (bull_power[i] > 0 and bull_power_roc[i] > 0 and bear_power_roc[i] < 0 and 
                adx_12h_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive AND improving (ROC > 0) AND Bull Power improving (ROC < 0, becoming less positive)
            # AND ADX > 25 (trending market) AND volume confirmation
            elif (bear_power[i] > 0 and bear_power_roc[i] > 0 and bull_power_roc[i] < 0 and 
                  adx_12h_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR Bear Power starts rising (ROC > 0, losing strength)
            if bull_power[i] <= 0 or bear_power_roc[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative OR Bull Power starts rising (ROC > 0, losing strength)
            if bear_power[i] <= 0 or bull_power_roc[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0