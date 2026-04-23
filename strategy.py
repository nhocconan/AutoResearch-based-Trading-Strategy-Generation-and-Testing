#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation.
- Camarilla pivot levels (R3/S3) act as strong intraday support/resistance; breakouts capture momentum.
- 12h EMA34 ensures alignment with intermediate trend to avoid counter-trend trades.
- Volume > 1.5x 20-period average confirms breakout validity.
- Discrete position size 0.25 limits drawdown during crashes.
- Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).
- Designed to work in both bull and bear regimes via trend filter and volume confirmation.
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
    
    # Calculate Camarilla pivot levels (R3, S3) from prior bar to avoid look-ahead
    # Use prior bar's OHLC to compute today's levels
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    pivot = (high_shift + low_shift + close_shift) / 3.0
    range_ = high_shift - low_shift
    r3 = pivot + range_ * 1.1 / 4.0  # R3 level
    s3 = pivot - range_ * 1.1 / 4.0  # S3 level
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Camarilla needs prior bar, 12h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close > R3 AND price above 12h EMA34 AND volume confirmation
            if close[i] > r3[i] and close[i] > ema_34_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < S3 AND price below 12h EMA34 AND volume confirmation
            elif close[i] < s3[i] and close[i] < ema_34_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < S3 OR price crosses below 12h EMA34
            if close[i] < s3[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > R3 OR price crosses above 12h EMA34
            if close[i] > r3[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0