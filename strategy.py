#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d Supertrend(10,3) filter and volume confirmation (>2.0x 20-period average).
# Long when price breaks above Donchian upper band AND 1d Supertrend = bullish AND volume > 2.0x MA20.
# Short when price breaks below Donchian lower band AND 1d Supertrend = bearish AND volume > 2.0x MA20.
# Exit when price crosses the Donchian midpoint (mean of upper/lower band).
# Uses 1d HTF Supertrend for strong trend filtering to reduce whipsaw and overtrading.
# Higher volume threshold (2.0x) and Donchian(20) structure target 75-200 trades over 4 years.
# Supertrend provides dynamic trend detection that adapts to volatility, working in both bull and bear markets.

name = "4h_Donchian20_Breakout_1dSupertrend10_3_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Volume confirmation: > 2.0x 20-period average (high threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Donchian Channel (20)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Supertrend(10, 3)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1d + low_1d) / 2.0
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, 1)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    start_idx = atr_period
    if start_idx < len(close_1d):
        supertrend[start_idx] = upper_band[start_idx]
        direction[start_idx] = 1
    
    # Calculate Supertrend
    for i in range(start_idx + 1, len(close_1d)):
        if close_1d[i-1] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
        
        # Reversal conditions
        if direction[i] == 1 and close_1d[i] < supertrend[i]:
            direction[i] = -1
            supertrend[i] = upper_band[i]
        elif direction[i] == -1 and close_1d[i] > supertrend[i]:
            direction[i] = 1
            supertrend[i] = lower_band[i]
    
    # Align Supertrend and direction to LTF
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(donchian_mid[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(supertrend_aligned[i]) or
            np.isnan(direction_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND 1d Supertrend uptrend AND volume confirm
            if (close[i] > high_20[i] and 
                direction_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band AND 1d Supertrend downtrend AND volume confirm
            elif (close[i] < low_20[i] and 
                  direction_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian midpoint (trend weakening)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian midpoint (trend weakening)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals