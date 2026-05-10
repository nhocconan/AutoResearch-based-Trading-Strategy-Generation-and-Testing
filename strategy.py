#!/usr/bin/env python3
# 4H_Camarilla_Pivot_Reversal_Scalp
# Hypothesis: Trade reversals at Camarilla pivot levels (S1/S3 for longs, R1/R3 for shorts) with volume confirmation and intraday trend filter.
# Works in both bull and bear markets by fading intraday extremes at statistically significant levels.
# Uses 1d trend filter to align with higher timeframe bias.
# Target: 25-40 trades/year per symbol.

name = "4H_Camarilla_Pivot_Reversal_Scalp"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    # Shift by 1 to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    S1 = prev_close - (range_ * 1.0 / 12)
    S2 = prev_close - (range_ * 2.0 / 12)
    S3 = prev_close - (range_ * 3.0 / 12)
    R1 = prev_close + (range_ * 1.0 / 12)
    R2 = prev_close + (range_ * 2.0 / 12)
    R3 = prev_close + (range_ * 3.0 / 12)
    
    # Align levels to 4h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Intraday trend filter: 9-period EMA vs 21-period EMA
    close_s = pd.Series(close)
    ema9 = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    intraday_uptrend = ema9 > ema21
    intraday_downtrend = ema9 < ema21
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S1_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(intraday_uptrend[i]) or 
            np.isnan(intraday_downtrend[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Long setup: price at S1 or S3, intraday uptrend, volume confirmation
            near_s1 = abs(close[i] - S1_aligned[i]) < (close[i] * 0.002)  # Within 0.2%
            near_s3 = abs(close[i] - S3_aligned[i]) < (close[i] * 0.002)
            if (near_s1 or near_s3) and intraday_uptrend[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short setup: price at R1 or R3, intraday downtrend, volume confirmation
            near_r1 = abs(close[i] - R1_aligned[i]) < (close[i] * 0.002)
            near_r3 = abs(close[i] - R3_aligned[i]) < (close[i] * 0.002)
            if (near_r1 or near_r3) and intraday_downtrend[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches S2 or intraday trend turns down
            near_s2 = abs(close[i] - S2_aligned[i]) < (close[i] * 0.002)
            if near_s2 or not intraday_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches R2 or intraday trend turns up
            near_r2 = abs(close[i] - R2_aligned[i]) < (close[i] * 0.002)
            if near_r2 or not intraday_downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals