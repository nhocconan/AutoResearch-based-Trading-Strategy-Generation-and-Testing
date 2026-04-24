#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Long when price breaks above 20-period Donchian high AND 1d ATR ratio > 1.2 (low volatility regime) AND volume > 1.5 * 20-period average volume
- Short when price breaks below 20-period Donchian low AND 1d ATR ratio > 1.2 (low volatility regime) AND volume > 1.5 * 20-period average volume
- ATR ratio = current 1d ATR(14) / 20-period average 1d ATR(14) - identifies low volatility breakout opportunities
- Exit on opposite Donchian level (exit long on Donchian low, exit short on Donchian high)
- Uses 4h primary with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian provides clear breakout levels; ATR regime filter avoids high volatility false breakouts; volume confirms momentum
- Designed to work in both bull (breakouts with momentum) and bear (breakdowns with momentum) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for ATR and its MA
        return np.zeros(n)
    
    # True Range calculation for 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar has no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - exponential moving average of True Range
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period average of ATR(14) for regime identification
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio: current ATR / average ATR (< 1.2 = low volatility regime)
    atr_ratio = atr_14 / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Low volatility regime filter: ATR ratio < 1.2
    low_vol_regime = atr_ratio_aligned < 1.2
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average (moderate spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # Need ATR MA and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(low_vol_regime[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND low volatility regime AND volume confirmation
            if close[i] > donchian_high[i] and low_vol_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND low volatility regime AND volume confirmation
            elif close[i] < donchian_low[i] and low_vol_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian low (opposite level)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian high (opposite level)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0