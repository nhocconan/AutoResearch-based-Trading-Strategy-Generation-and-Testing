#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Long when price breaks above Donchian(20) high AND 1d ATR(14) > 1.5 * 1d ATR(50) (expanding volatility)
- Short when price breaks below Donchian(20) low AND 1d ATR(14) > 1.5 * 1d ATR(50) (expanding volatility)
- Volume confirmation: current volume > 2.0 * 20-period average volume (strong spike)
- Exit on opposite Donchian level (L20 for long exit, H20 for short exit)
- Uses 4h primary with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Donchian provides price channel structure; ATR filter ensures breakouts occur in expanding volatility; volume spike confirms momentum
- Designed to work in both bull (breakouts with trend) and bear (breakouts against trend) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Donchian(20) channels (based on previous 20 bars to avoid look-ahead)
    # Upper = max(high of previous 20 bars)
    # Lower = min(low of previous 20 bars)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_h20 = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_l20 = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility expansion filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility expansion: ATR(14) > 1.5 * ATR(50)
    vol_expansion = atr_14 > (1.5 * atr_50)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50, 20) + 1  # Need Donchian(20), ATRs, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h20[i]) or np.isnan(donchian_l20[i]) or 
            np.isnan(vol_expansion_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian H20 AND volatility expansion AND volume confirmation
            if close[i] > donchian_h20[i] and vol_expansion_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian L20 AND volatility expansion AND volume confirmation
            elif close[i] < donchian_l20[i] and vol_expansion_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian L20 (opposite level)
            if close[i] < donchian_l20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian H20 (opposite level)
            if close[i] > donchian_h20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRFilter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0