#!/usr/bin/env python3
"""
Hypothesis: 1h Williams %R mean reversion with 4h EMA50 trend filter and volume spike confirmation.
- Uses 1h timeframe (primary) and 4h HTF for EMA50 trend alignment
- Williams %R(14) measures overbought/oversold: long when %R crosses above -80 from below, short when crosses below -20 from above
- Trend filter: only long when price > 4h EMA50, only short when price < 4h EMA50
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when %R crosses midpoint (-50) for mean reversion
- Discrete signal size: 0.20 to minimize fee churn
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Williams %R captures mean reversion in all regimes
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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 4h EMA50
    uptrend = close > ema_50_4h_aligned
    downtrend = close < ema_50_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, lookback)  # Need 4h EMA50, volume MA(20), and Williams %R lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND uptrend AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Williams %R crosses below -20 from above AND downtrend AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (mean reversion) or reverse signal
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (mean reversion) or reverse signal
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_WilliamsR14_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0