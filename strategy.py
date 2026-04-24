#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA34 trend alignment
- Donchian levels calculated from prior 20-period 12h candles: upper = max(high), lower = min(low)
- Breakout logic: long when price crosses above upper band with volume confirmation, short when price crosses below lower band
- Trend filter: only long when price > 1d EMA34, only short when price < 1d EMA34
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to prior 12h close (mean reversion)
- Discrete signal size: 0.25 to balance return and risk
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior 20-period 12h Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_12h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_12h, donch_lower)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    # Mean reversion exit: price reverts to prior 12h close
    df_12h_close = df_12h['close'].values
    prev_close_12h = pd.Series(df_12h_close).shift(1).values
    prev_close_12h_aligned = align_htf_to_ltf(prices, df_12h, prev_close_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1d EMA34 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(prev_close_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above upper band AND uptrend AND volume confirmation
            if close[i] > donch_upper_aligned[i] and close[i-1] <= donch_upper_aligned[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower band AND downtrend AND volume confirmation
            elif close[i] < donch_lower_aligned[i] and close[i-1] >= donch_lower_aligned[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 12h close (mean reversion) or reverse signal
            if close[i] <= prev_close_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 12h close (mean reversion) or reverse signal
            if close[i] >= prev_close_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0