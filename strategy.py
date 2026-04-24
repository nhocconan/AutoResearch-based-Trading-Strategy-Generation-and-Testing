#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator strategy with 1d trend filter and volume confirmation.
- Uses Williams Alligator (Jaw=TEETH=LIPS SMMA) to identify trend direction and entry timing.
- Long when Lips cross above Jaw with price above 1d EMA50 and volume > 2x average.
- Short when Lips cross below Jaw with price below 1d EMA50 and volume > 2x average.
- Uses discrete position size 0.25 to manage drawdown and reduce fee churn.
- Designed for 4h timeframe to capture medium-term trends in both bull and bear markets.
- Alligator provides natural smoothing and reduces false signals vs. simple MA crossovers.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (Williams Alligator uses SMMA)"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 4h data
    # Jaw: SMMA(13, 8) - blue line
    # Teeth: SMMA(8, 5) - red line  
    # Lips: SMMA(5, 3) - green line
    jaw = smma(close, 13)
    jaw = smma(jaw, 8)  # SMMA of SMMA with offset 8
    teeth = smma(close, 8)
    teeth = smma(teeth, 5)  # SMMA of SMMA with offset 5
    lips = smma(close, 5)
    lips = smma(lips, 3)  # SMMA of SMMA with offset 3
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Lips cross above Jaw AND price above 1d EMA50 AND volume confirmation
            if lips[i-1] <= jaw[i-1] and lips[i] > jaw[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Lips cross below Jaw AND price below 1d EMA50 AND volume confirmation
            elif lips[i-1] >= jaw[i-1] and lips[i] < jaw[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Lips cross below Jaw OR price below 1d EMA50
            if lips[i-1] >= jaw[i-1] and lips[i] < jaw[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips cross above Jaw OR price above 1d EMA50
            if lips[i-1] <= jaw[i-1] and lips[i] > jaw[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0