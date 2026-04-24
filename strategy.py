#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation.
- Camarilla pivot levels (H3, L3) calculated from previous 1d bar: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
- Long when price breaks above H3 AND price > 1d EMA50 (uptrend filter) AND volume > 1.5 * volume MA20
- Short when price breaks below L3 AND price < 1d EMA50 (downtrend filter) AND volume > 1.5 * volume MA20
- Exit when price reverts to Camarilla H4/L4 levels
- Designed to capture institutional breakout attempts with trend alignment and volume confirmation
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Calculate 1d OHLC for Camarilla pivots (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    # Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla H3 and L3 (using previous bar's OHLC)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 12h timeframe (previous day's levels available at open)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.5 * volume MA20
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and downtrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Camarilla H4 level
            camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2  # H4 level
            camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            if close[i] < camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Camarilla L4 level
            camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2  # L4 level
            camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            if close[i] > camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0