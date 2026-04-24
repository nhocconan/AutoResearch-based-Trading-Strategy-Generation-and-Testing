#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume confirmation.
- Uses 12h timeframe for HTF trend alignment (more stable than 1d for 4h entries)
- Camarilla H3/L3 from previous 12h bar: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
- Long when price breaks above H3 AND price > 12h EMA50 (uptrend) AND volume > 1.5 * volume MA(20)
- Short when price breaks below L3 AND price < 12h EMA50 (downtrend) AND volume > 1.5 * volume MA(20)
- Exit when price reverts to Camarilla H4/L4 levels
- Discrete signal size: 0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h OHLC for Camarilla pivots (using previous completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    # Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla H3 and L3 (using previous bar's OHLC)
    camarilla_h3 = close_12h + 1.1 * (high_12h - low_12h) / 4
    camarilla_l3 = close_12h - 1.1 * (high_12h - low_12h) / 4
    
    # Align Camarilla levels to 4h timeframe (previous 12h bar's levels available at open)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 12h EMA50
    uptrend = close > ema_50_12h_aligned
    downtrend = close < ema_50_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 12h EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Camarilla H4 level
            camarilla_h4 = close_12h + 1.1 * (high_12h - low_12h) / 2  # H4 level
            camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
            if close[i] < camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Camarilla L4 level
            camarilla_l4 = close_12h - 1.1 * (high_12h - low_12h) / 2  # L4 level
            camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
            if close[i] > camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0