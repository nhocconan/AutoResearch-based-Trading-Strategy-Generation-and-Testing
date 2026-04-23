#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13. Combined with weekly trend filter to avoid
counter-trend trades and volume confirmation for breakout validity. Designed to work in both
bull and bear markets by following the higher timeframe trend. Target: 12-37 trades/year per symbol.
Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate EMA13 for Elder Ray (on 6h timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_13_6h)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6h_aligned
    bear_power = low - ema_13_6h_aligned
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)  # need EMA50(1w), EMA13(6h), and vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_13_6h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1w EMA50 = uptrend, close < 1w EMA50 = downtrend
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND uptrend AND volume confirmation
            if bull_power[i] > 0 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND downtrend AND volume confirmation
            elif bear_power[i] < 0 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite Elder Ray signal (Bear Power >= 0 for longs, Bull Power <= 0 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long when bear power becomes non-negative (weakening bullish momentum)
                if bear_power[i] >= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short when bull power becomes non-positive (weakening bearish momentum)
                if bull_power[i] <= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1wEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0