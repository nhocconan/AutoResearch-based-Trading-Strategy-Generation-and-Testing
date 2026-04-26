#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirmation
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter (EMA50) and volume confirmation (>1.8x 20-period MA).
Long when Bear Power < 0 (bulls in control) AND price > 1d EMA50 (uptrend) AND volume spike.
Short when Bull Power > 0 (bears in control) AND price < 1d EMA50 (downtrend) AND volume spike.
Elder Ray measures bull/bear strength relative to EMA13, filtering weak moves. Works in both bull/bear markets by following 1d trend.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year on 6h timeframe.
"""

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
    
    # Get 1d data for trend filter (prior completed daily candle)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power: measures bull strength
    bear_power = low - ema_13   # Bear Power: measures bear strength (negative = bulls in control)
    
    # Volume confirmation: volume > 1.8x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 13 for EMA13)
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: Bear Power < 0 (bulls in control) + 1d uptrend + volume spike
            if (bear_power[i] < 0 and uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power > 0 (bears in control) + 1d downtrend + volume spike
            elif (bull_power[i] > 0 and downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bear Power becomes positive (bears take control) OR 1d trend changes to downtrend
            if (bear_power[i] >= 0 or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power becomes negative (bulls take control) OR 1d trend changes to uptrend
            if (bull_power[i] <= 0 or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0