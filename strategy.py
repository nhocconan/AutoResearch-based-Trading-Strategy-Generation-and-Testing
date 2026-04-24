#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for Camarilla levels, EMA trend, and volume confirmation.
- Camarilla levels (R3, S3) calculated from 1d OHLC: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2.
- Entry: Long when price breaks above R3 with volume spike and close > 1d EMA34 (uptrend).
         Short when price breaks below S3 with volume spike and close < 1d EMA34 (downtrend).
- Exit: When price reverts to the 1d close (pivot level) or opposite signal.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, EMA trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = df_1d['close'].values + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 2
    camarilla_s3 = df_1d['close'].values - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 2
    
    # Calculate 1d volume for confirmation (using 1d volume MA)
    volume_1d_ma = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # Volume confirmation: current 4h volume > 2.0 * aligned 1d volume MA
    volume_spike = volume > (2.0 * volume_1d_ma_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout with volume spike and trend filter
            if volume_spike[i]:
                # Long: price breaks above R3 with uptrend
                if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S3 with downtrend
                elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to 1d close (pivot) or breaks below S3
            if close[i] < df_1d['close'].values[-1] if i == len(prices)-1 else camarilla_s3_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to 1d close (pivot) or breaks above R3
            if close[i] > df_1d['close'].values[-1] if i == len(prices)-1 else camarilla_r3_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0