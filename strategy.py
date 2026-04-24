#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for ADX trend alignment
- Donchian channels calculated from prior 20-period 12h high/low: upper = max(high[-20:]), lower = min(low[-20:])
- Breakout logic: long when price closes above upper band with volume spike and uptrend (ADX>25),
                  short when price closes below lower band with volume spike and downtrend (ADX>25)
- Trend filter: 1d ADX > 25 indicates strong trend (avoids choppy markets)
- Volume confirmation: current 12h volume > 1.5 * 20-period 12h volume MA
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
- Works in both bull/bear: trend filter avoids false breakouts in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high_1d[0] - low_1d[0]  # First period TR
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    up_move.iloc[0] = 0
    down_move.iloc[0] = 0
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need Donchian(20) and 1d ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND ADX>25 (uptrend) AND volume spike
            if close[i] > donchian_upper[i] and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band AND ADX>25 (downtrend) AND volume spike
            elif close[i] < donchian_lower[i] and adx_aligned[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channel or reverse signal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] <= donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channel or reverse signal
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] >= donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0