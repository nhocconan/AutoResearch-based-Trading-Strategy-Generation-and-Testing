#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation.
- Uses 4h timeframe (primary) and 12h HTF for EMA34 trend alignment (more responsive than 1d)
- Camarilla levels calculated from prior 4h OHLC: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
- Breakout logic: long when price crosses above H3 with volume confirmation, short when price crosses below L3
- Trend filter: only long when price > 12h EMA34, only short when price < 12h EMA34
- Volume confirmation: current volume > 1.8 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to prior 4h close (mean reversion)
- Discrete signal size: 0.25 to balance return and risk
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
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
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate prior 4h Camarilla levels (H3 and L3) - using 4h HTF for level calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h)
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h)
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * volume_ma)
    
    # Trend filter: price above/below 12h EMA34
    uptrend = close > ema_34_12h_aligned
    downtrend = close < ema_34_12h_aligned
    
    # Mean reversion exit: price reverts to prior 4h close
    prev_close_4h = df_4h['close'].shift(1).values
    prev_close_aligned = align_htf_to_ltf(prices, df_4h, prev_close_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 12h EMA34 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(prev_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above H3 AND uptrend AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 4h close (mean reversion) or reverse signal
            if close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 4h close (mean reversion) or reverse signal
            if close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0