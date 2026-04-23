#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
- Camarilla levels calculated from prior 1d OHLC (R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low))
- Long: price breaks above R3 + volume > 2x 20-period avg + price > 1w EMA50 (uptrend)
- Short: price breaks below S3 + volume > 2x 20-period avg + price < 1w EMA50 (downtrend)
- Exit: price reverts to prior day's close (mean reversion to equilibrium)
- 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag on 1d timeframe
- Works in both bull (trend continuation via breakouts) and bear (mean reversion via Camarilla equilibrium)
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
    open_time = prices['open_time'].values
    
    # Volume confirmation: > 2x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Camarilla levels and prior close
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from prior 1d OHLC
    # R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    prior_close = close_1d  # equilibrium level for mean reversion exit
    
    # Align HTF arrays to LTF (1d values available only after daily bar closes)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(prior_close_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + price > 1w EMA50 (uptrend)
            if volume_spike and close[i] > camarilla_r3_aligned[i]:
                if close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S3 + volume spike + price < 1w EMA50 (downtrend)
            elif volume_spike and close[i] < camarilla_s3_aligned[i]:
                if close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to prior day's close (mean reversion to equilibrium)
            if close[i] <= prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior day's close (mean reversion to equilibrium)
            if close[i] >= prior_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0