#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend.
- Camarilla pivot levels (R3, S3) from prior 1d: Long when price > R3, Short when price < S3.
- Trend filter: Only trade in direction of 12h EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 1.8x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Uses actual Camarilla calculation: R3 = H + 1.1*(L-C), S3 = L - 1.1*(H-C).
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # R3 = H + 1.1*(L - C), S3 = L - 1.1*(H - C)
    camarilla_R3 = high_1d + 1.1 * (low_1d - close_1d)
    camarilla_S3 = low_1d - 1.1 * (high_1d - close_1d)
    
    # Align to 4h: use prior 1d's levels (already completed bar)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 12h EMA50 trend
            if i > 0 and not np.isnan(ema_50_12h_aligned[i-1]):
                ema50_slope = ema_50_12h_aligned[i] - ema_50_12h_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    if close[i] > camarilla_R3_aligned[i] and volume_spike[i]:
                        # Buy on R3 breakout in uptrend
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    if close[i] < camarilla_S3_aligned[i] and volume_spike[i]:
                        # Sell on S3 breakdown in downtrend
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price returns to prior 1d close or opposite break
            if close[i] < camarilla_S3_aligned[i] or close[i] < camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to prior 1d close or opposite break
            if close[i] > camarilla_R3_aligned[i] or close[i] > camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0