#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend direction.
- Entry: Long when price breaks above Camarilla R3 AND 12h EMA50 > EMA200 (bullish trend).
         Short when price breaks below Camarilla S3 AND 12h EMA50 < EMA200 (bearish trend).
         Volume confirmation: current volume > 1.5 * 20-period volume MA to filter false breakouts.
- Exit: Opposite Camarilla level break (R3/S3) or EMA trend flip.
- Uses discrete signal size 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull/bear via trend filter: only takes breakouts in direction of 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 and EMA200
    close_12h = pd.Series(df_12h['close'])
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = close_12h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3, R4, S3, S4
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align 1d Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 200, 20)  # Need enough 12h bars for EMA200 and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above R3 AND 12h EMA50 > EMA200 (uptrend)
                if curr_high > r3_aligned[i] and ema50_12h_aligned[i] > ema200_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S3 AND 12h EMA50 < EMA200 (downtrend)
                elif curr_low < s3_aligned[i] and ema50_12h_aligned[i] < ema200_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR 12h EMA50 < EMA200 (trend flip)
            if curr_low < s3_aligned[i] or ema50_12h_aligned[i] < ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR 12h EMA50 > EMA200 (trend flip)
            if curr_high > r3_aligned[i] or ema50_12h_aligned[i] > ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_12hEMATrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0