# -*- coding: utf-8 -*-
# -*- mode: python; -*-

#!/usr/bin/env python3
"""
Hypothesis:
    - 6h timeframe reduces noise vs lower timeframes while capturing meaningful swings.
    - Use 12h timeframe for trend direction (via EMA crossover) to avoid whipsaws.
    - Enter on 6h retracements to the 12h EMA(21) with volume confirmation.
    - Exit when price crosses the 12h EMA(50) in the opposite direction.
    - This combines trend-following (12h EMA21/50) with mean-reversion entries (6h pullback).
    - Designed to work in both bull and bear markets by following the 12h trend.
    - Volume filter ensures momentum behind moves, reducing false signals.
    - Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data once for trend and filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(21) for trend entry level
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 12h EMA(50) for exit condition
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h ATR(14) for volume filter normalization (optional)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_12h = np.full(len(tr_12h), np.nan)
    for i in range(14, len(tr_12h)):
        atr_12h[i] = np.nanmean(tr_12h[i-13:i+1])
    
    # Align 12h indicators to 6h timeframe
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volume filter: use 6h volume vs its 20-period moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital per trade
    
    # Start loop after sufficient warmup for indicators
    start_idx = max(50, 20)  # cover EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_21_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average
        volume_ok = volume[i] > 1.5 * vol_ma_20[i]
        
        # Long conditions: price near 12h EMA21 from below, in uptrend (EMA21 > EMA50), volume confirmation
        if ema_21_12h_aligned[i] > ema_50_12h_aligned[i]:  # uptrend filter
            if position == 0:
                # Enter long: price crosses above EMA21 with volume
                if close[i] > ema_21_12h_aligned[i] and close[i-1] <= ema_21_12h_aligned[i-1] and volume_ok:
                    position = 1
                    signals[i] = position_size
            elif position == 1:
                # Exit long: price crosses below EMA50 (trend weakening)
                if close[i] < ema_50_12h_aligned[i] and close[i-1] >= ema_50_12h_aligned[i-1]:
                    position = 0
                    signals[i] = 0.0
        
        # Short conditions: price near 12h EMA21 from above, in downtrend (EMA21 < EMA50), volume confirmation
        elif ema_21_12h_aligned[i] < ema_50_12h_aligned[i]:  # downtrend filter
            if position == 0:
                # Enter short: price crosses below EMA21 with volume
                if close[i] < ema_21_12h_aligned[i] and close[i-1] >= ema_21_12h_aligned[i-1] and volume_ok:
                    position = -1
                    signals[i] = -position_size
            elif position == -1:
                # Exit short: price crosses above EMA50 (trend weakening)
                if close[i] > ema_50_12h_aligned[i] and close[i-1] <= ema_50_12h_aligned[i-1]:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_12hEMA21_50_VolumePullback"
timeframe = "6h"
leverage = 1.0