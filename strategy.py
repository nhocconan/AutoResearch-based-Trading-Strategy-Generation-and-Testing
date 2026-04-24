#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
- Williams %R(14) identifies overbought/oversold conditions; reversals from extremes provide edge.
- 1d EMA50 filter ensures trades align with higher-timeframe trend, reducing counter-trend whipsaws.
- Volume confirmation (>2.0x 24-period average) ensures conviction on breakouts/reversals.
- Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years), balancing opportunity and fee drag.
- Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.
- Works in both bull and bear markets: %R reversals capture mean reversion in ranges, while EMA filter avoids fighting strong trends.
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
    
    # Prior 1d OHLC (completed daily bar)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Align to 6h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14) on 6h
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    # Avoid division by zero
    denom = highest_high - lowest_low
    denom = np.where(denom == 0, 1e-10, denom)
    willr = -100 * (highest_high - close) / denom
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, period, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND price above 1d EMA50 AND volume confirmation
            if willr[i] > -80 and willr[i-1] <= -80 and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND price below 1d EMA50 AND volume confirmation
            elif willr[i] < -20 and willr[i-1] >= -20 and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR price crosses below 1d EMA50
            if willr[i] < -50 and willr[i-1] >= -50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR price crosses above 1d EMA50
            if willr[i] > -50 and willr[i-1] <= -50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0