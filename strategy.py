#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 6h timeframe (primary) and 1d HTF for EMA34 trend alignment
- Camarilla levels calculated from prior 1d OHLC: R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
- Breakout logic: long when price closes above R3 with volume spike and uptrend,
                  short when price closes below S3 with volume spike and downtrend
- Trend filter: only long when 6h EMA21 > 1d EMA34, only short when 6h EMA21 < 1d EMA34
- Volume confirmation: current 6h volume > 2.0 * 20-period 6h volume MA to capture institutional interest
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
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
    
    # Calculate 6h EMA21 for trend confirmation (faster than EMA34)
    ema_21_6h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior 1d Camarilla levels (R3, S3)
    # Need to shift 1d data by 1 to avoid look-ahead (use prior completed day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's Camarilla: R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align to 6h timeframe (wait for 1d bar to close)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 6h EMA21 vs 1d EMA34
    uptrend = ema_21_6h > ema_34_1d_aligned
    downtrend = ema_21_6h < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 21)  # Need 1d EMA34 and 6h EMA21
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above R3 AND uptrend AND volume spike
            if close[i] > camarilla_r3_1d_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below S3 AND downtrend AND volume spike
            elif close[i] < camarilla_s3_1d_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 1d S4 (mean reversion) or reverse signal
            camarilla_s4_1d = close_1d - 1.1 * (high_1d - low_1d)
            camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
            if not np.isnan(camarilla_s4_1d_aligned[i]) and close[i] <= camarilla_s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 1d R4 (mean reversion) or reverse signal
            camarilla_r4_1d = close_1d + 1.1 * (high_1d - low_1d)
            camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
            if not np.isnan(camarilla_r4_1d_aligned[i]) and close[i] >= camarilla_r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0