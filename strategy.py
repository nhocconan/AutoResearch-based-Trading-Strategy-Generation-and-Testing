#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index combined with 1-week EMA50 trend filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA)
- Long when Bull Power > 0 AND Bear Power < 0 AND price > weekly EMA50 AND volume > 1.5x average
- Short when Bear Power > 0 AND Bull Power < 0 AND price < weekly EMA50 AND volume > 1.5x average
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Volume confirmation requires >1.5x 24-period average to ensure conviction.
- Designed for 12-30 trades/year (50-120 total over 4 years) to stay within fee-efficient range.
- Combines proven elements: Elder Ray momentum + weekly trend filter + volume confirmation.
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior 1w close (completed weekly bar)
    close_1w = df_1w['close'].shift(1).values
    
    # Align to 6h timeframe
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Elder Ray components (using 13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 24, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > weekly EMA50 AND volume confirmation
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND price < weekly EMA50 AND volume confirmation
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR price crosses below weekly EMA50
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 OR Bull Power >= 0 OR price crosses above weekly EMA50
            if bear_power[i] <= 0 or bull_power[i] >= 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0