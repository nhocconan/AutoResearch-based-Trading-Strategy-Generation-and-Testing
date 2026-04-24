#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation.
- Elder Ray measures bull/bear power relative to EMA13 to detect conviction.
- Uses 1d EMA50 for higher timeframe trend alignment (works in bull/bear).
- Volume confirmation (>1.5x 20-bar average) ensures conviction.
- Discrete position size 0.25 limits drawdown and reduces fee churn.
- Targets 12-30 trades/year (50-120 total over 4 years) to stay fee-efficient.
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d close (completed 1d bar)
    close_1d = df_1d['close'].shift(1).values
    
    # Align 1d EMA50 to 6h
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price above 1d EMA50 AND volume confirmation
            if bull_power[i] > 0 and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price below 1d EMA50 AND volume confirmation
            elif bear_power[i] < 0 and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR price crosses below 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR price crosses above 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0