#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot reversal with volume confirmation and 12h EMA trend filter.
- Camarilla levels (S1/S2/S3, R1/R2/R3) act as strong support/resistance zones
- Reversal at S1/R1 with volume spike indicates institutional interest
- 12h EMA50 filters for trend direction, avoids counter-trend trades
- Exit on opposite Camarilla level (S3/R3) or trend reversal
- Target: 20-30 trades/year to avoid fee drag
- Uses discrete position sizing (0.25) to minimize churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close
    close_val = close
    S1 = close - (range_val * 1.0 / 12)
    S2 = close - (range_val * 2.0 / 12)
    S3 = close - (range_val * 3.0 / 12)
    R1 = close + (range_val * 1.0 / 12)
    R2 = close + (range_val * 2.0 / 12)
    R3 = close + (range_val * 3.0 / 12)
    return S1, S2, S3, R1, R2, R3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    S1_1d = np.full(len(high_1d), np.nan)
    S2_1d = np.full(len(high_1d), np.nan)
    S3_1d = np.full(len(high_1d), np.nan)
    R1_1d = np.full(len(high_1d), np.nan)
    R2_1d = np.full(len(high_1d), np.nan)
    R3_1d = np.full(len(high_1d), np.nan)
    
    for i in range(len(high_1d)):
        S1, S2, S3, R1, R2, R3 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        S1_1d[i] = S1
        S2_1d[i] = S2
        S3_1d[i] = S3
        R1_1d[i] = R1
        R2_1d[i] = R2
        R3_1d[i] = R3
    
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    for i in range(len(close_12h)):
        if i >= 49:  # 50-period EMA
            if i == 49:
                ema_50_12h[i] = np.mean(close_12h[:50])
            else:
                ema_50_12h[i] = (close_12h[i] * 2 / 51) + (ema_50_12h[i-1] * 49 / 51)
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 60)
    
    for i in range(start_idx, n):
        if (np.isnan(S1_1d_aligned[i]) or np.isnan(R1_1d_aligned[i]) or
            np.isnan(S3_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price at S1 with volume spike + uptrend (price > 12h EMA50)
            if (abs(close[i] - S1_1d_aligned[i]) < (S1_1d_aligned[i] - S2_1d_aligned[i]) * 0.5 and
                volume_spike[i] and
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price at R1 with volume spike + downtrend (price < 12h EMA50)
            elif (abs(close[i] - R1_1d_aligned[i]) < (R2_1d_aligned[i] - R1_1d_aligned[i]) * 0.5 and
                  volume_spike[i] and
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches S3 (strong support) OR trend reversal
            if (close[i] <= S3_1d_aligned[i] or
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R3 (strong resistance) OR trend reversal
            if (close[i] >= R3_1d_aligned[i] or
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S1R1_VolumeSpike_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0