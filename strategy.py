#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
- Uses 1d Camarilla pivot levels (R3, S3, R4, S4) calculated from prior 1d OHLC
- Long breakout: price > R3 + volume > 1.5x 20-period avg + price > 1d EMA50 (uptrend)
- Short breakdown: price < S3 + volume > 1.5x 20-period avg + price < 1d EMA50 (downtrend)
- Continuation breakout at R4/S4 with same filters for stronger moves
- Exit: price reverts to prior 1d close (mean reversion to fair value)
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
- Volume confirmation reduces false breakouts in low-participation moves
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Camarilla pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # R4 = close + 1.5*(high - low)
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # S4 = close - 1.5*(high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_middle = close_1d  # pivot point = prior close
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_middle_aligned = align_htf_to_ltf(prices, df_1d, camarilla_middle)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_middle_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > R3 + volume spike + price > 1d EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Strong continuation breakout at R4
                elif close[i] > camarilla_r4_aligned[i]:
                    signals[i] = 0.30
                    position = 1
            # Short breakdown: price < S3 + volume spike + price < 1d EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                # Strong continuation breakdown at S4
                elif close[i] < camarilla_s4_aligned[i]:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long exit: price reverts to prior 1d close (mean reversion)
            if close[i] <= camarilla_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if close[i] < camarilla_r4_aligned[i] else 0.30
        elif position == -1:
            # Short exit: price reverts to prior 1d close (mean reversion)
            if close[i] >= camarilla_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 if close[i] > camarilla_s4_aligned[i] else -0.30
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0