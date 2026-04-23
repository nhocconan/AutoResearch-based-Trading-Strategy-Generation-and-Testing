#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Breakout with 1d EMA50 Trend Filter and Volume Spike Confirmation.
- Primary timeframe: 6h, HTF: 1d for trend filter
- Camarilla pivot levels calculated from prior 1d OHLC
- Long: Close breaks above R3 + price > 1d EMA50 (uptrend) + volume > 2.0x 20-period avg
- Short: Close breaks below S3 + price < 1d EMA50 (downtrend) + volume > 2.0x 20-period avg
- Exit: Close reverts to 1d EMA50 (mean reversion to trend)
- Uses tight Camarilla R3/S3 levels for high-probability breakouts in both bull and bear markets
- Volume spike filter reduces false breakouts
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Volume confirmation: > 2.0x 20-period average (strong volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Camarilla pivot and EMA50
    df_1d = get_htf_data(prices, '1d')
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d
    camarilla_s3 = close_1d - 1.1 * range_1d
    camarilla_r4 = close_1d + 1.5 * range_1d
    camarilla_s4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R3 + price > 1d EMA50 (uptrend) + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + price < 1d EMA50 (downtrend) + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close reverts to or below 1d EMA50 (mean reversion to trend)
            if close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close reverts to or above 1d EMA50 (mean reversion to trend)
            if close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0