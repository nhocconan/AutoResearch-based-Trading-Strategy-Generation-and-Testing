#!/usr/bin/env python3
# 12H_Camarilla_Pivot_Reversal_1DTrend
# Hypothesis: Trade reversals at Camarilla pivot levels on 12h timeframe with 1d trend filter.
# Long when: price touches or crosses below Camarilla S3 level in 1d uptrend with volume spike.
# Short when: price touches or crosses above Camarilla R3 level in 1d downtrend with volume spike.
# Uses 12h volume confirmation and exit on opposite touch of R1/S1.
# Designed for low trade frequency (~20-40/year) with high win rate in trending and ranging markets.
# Works in bull/bear by following 1d trend and using mean-reversion at extreme pivots.

name = "12H_Camarilla_Pivot_Reversal_1DTrend"
timeframe = "12h"
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
    
    # 12h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Calculate Camarilla levels from previous day's OHLC
    # We'll use rolling window of previous day's data
    high_prev = df_1d['high'].shift(1).values  # Previous day's high
    low_prev = df_1d['low'].shift(1).values    # Previous day's low
    close_prev = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels for each day
    R4 = close_prev + (high_prev - low_prev) * 1.5000
    R3 = close_prev + (high_prev - low_prev) * 1.2500
    R2 = close_prev + (high_prev - low_prev) * 1.1666
    R1 = close_prev + (high_prev - low_prev) * 1.0833
    PP = (high_prev + low_prev + close_prev) / 3.0
    S1 = close_prev - (high_prev - low_prev) * 1.0833
    S2 = close_prev - (high_prev - low_prev) * 1.1666
    S3 = close_prev - (high_prev - low_prev) * 1.2500
    S4 = close_prev - (high_prev - low_prev) * 1.5000
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0  # Require strong volume spike
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price at or below S3 in 1d uptrend with volume spike
            if daily_up and volume_confirm and low[i] <= S3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price at or above R3 in 1d downtrend with volume spike
            elif daily_down and volume_confirm and high[i] >= R3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches S1 (mean reversion target) or trend changes
            if high[i] >= S1_aligned[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches R1 (mean reversion target) or trend changes
            if low[i] <= R1_aligned[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals