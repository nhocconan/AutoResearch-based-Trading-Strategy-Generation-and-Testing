#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, fade extreme Camarilla R3/S3 levels from 1d with 1d trend filter and volume confirmation.
In ranging markets, price reverts from R3/S3; in trending markets, 1d trend filter prevents counter-trend fades.
Works in both bull/bear: fades provide liquidity capture in ranges, trend filter avoids whipsaws in strong moves.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d trend filter: EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 1.8x 24-period MA (4h equivalent on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 24 for volume MA + 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long fade: price drops to S3 level, with 1d uptrend and volume spike
            if low[i] <= camarilla_s3_aligned[i] and uptrend_1d[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short fade: price rises to R3 level, with 1d downtrend and volume spike
            elif high[i] >= camarilla_r3_aligned[i] and downtrend_1d[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price reverts to midpoint OR 1d trend changes to downtrend
            pivot_point = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] >= pivot_point or not uptrend_1d[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price reverts to midpoint OR 1d trend changes to uptrend
            pivot_point = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] <= pivot_point or not downtrend_1d[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0