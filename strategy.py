#!/usr/bin/env python3
# 1h_TripleConfirmation_RangeBound_MeanReversion
# Hypothesis: In range-bound markets (common in 2025-2026), price reverts to mean at Bollinger Bands extremes.
# Uses 4h trend filter (EMA50) to avoid counter-trend trades, and 1d volume spike for confirmation.
# Entry: Price touches Bollinger Band (20,2) AND 4h EMA50 filter aligned AND 1d volume > 1.5x 20-day average.
# Exit: Price returns to Bollinger middle band.
# Timeframe: 1h for precise entry/exit, 4h for trend filter, 1d for volume confirmation.
# Target: 20-40 trades/year with strict criteria to minimize fee drag.

name = "1h_TripleConfirmation_RangeBound_MeanReversion"
timeframe = "1h"
leverage = 1.0

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
    
    # Bollinger Bands (20, 2)
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = ma + 2.0 * std
    lower = ma - 2.0 * std
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at lower Bollinger Band, above 4h EMA50 (uptrend bias), high volume
            if close[i] <= lower[i] and close[i] > ema_4h_aligned[i] and volume[i] > 1.5 * vol_ma_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price at upper Bollinger Band, below 4h EMA50 (downtrend bias), high volume
            elif close[i] >= upper[i] and close[i] < ema_4h_aligned[i] and volume[i] > 1.5 * vol_ma_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price returns to middle band or trend bias lost
            if close[i] >= ma[i] or close[i] <= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price returns to middle band or trend bias lost
            if close[i] <= ma[i] or close[i] >= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals