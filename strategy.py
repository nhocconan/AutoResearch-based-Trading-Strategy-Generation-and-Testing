#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 Breakout with 12h EMA50 Trend Filter and Volume Confirmation.
- Camarilla pivots calculated from previous 12h bar (HLC of completed 12h bar)
- Long: Close breaks above R3 with volume > 1.5x 20-period average AND price > 12h EMA50
- Short: Close breaks below S3 with volume > 1.5x 20-period average AND price < 12h EMA50
- Exit: Close crosses back below R3 (for longs) or above S3 (for shorts) OR EMA50 trend fails
- Uses Camarilla for intraday support/resistance, EMA50 for HTF trend, volume for confirmation
- Target: 75-200 total trades over 4 years (19-50/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous completed 12h bar
    # Typical price = (H+L+C)/3
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    tp = typical_price_12h.values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Camarilla width = (High - Low) * 1.1 / 12
    camarilla_width = (high_12h - low_12h) * 1.1 / 12
    
    # R3 = TP + camarilla_width * 1.1
    # S3 = TP - camarilla_width * 1.1
    r3 = tp + camarilla_width * 1.1
    s3 = tp - camarilla_width * 1.1
    
    # Align Camarilla levels to 6h timeframe (using previous 12h bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R3 + volume confirmation + price > 12h EMA50
            if (close[i] > r3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + volume confirmation + price < 12h EMA50
            elif (close[i] < s3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses back below R3 OR price < 12h EMA50
            if close[i] < r3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses back above S3 OR price > 12h EMA50
            if close[i] > s3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0