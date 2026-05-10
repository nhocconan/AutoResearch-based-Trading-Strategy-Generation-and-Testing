#!/usr/bin/env python3
# 6h_Pivot_Reversal_Volume_Weighted
# Hypothesis: Price reversals at daily pivot points (S1/R1) with volume confirmation
# work in both bull and bear markets. In uptrends, buy at S1 with volume; in downtrends,
# sell at R1 with volume. Uses volume-weighted average price (VWAP) for dynamic
# support/resistance and avoids false breakouts. Targets 15-30 trades/year.

name = "6h_Pivot_Reversal_Volume_Weighted"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot_point = (daily_high + daily_low + daily_close) / 3
    daily_r1 = 2 * pivot_point - daily_low
    daily_s1 = 2 * pivot_point - daily_high
    
    # Align daily pivot levels to 6h timeframe
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Volume-weighted average price (VWAP) for dynamic support/resistance
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, 0)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(daily_r1_aligned[i]) or 
            np.isnan(daily_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price at or below S1 with volume confirmation
            if close[i] <= daily_s1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price at or above R1 with volume confirmation
            elif close[i] >= daily_r1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above VWAP or reversal signal
            if close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below VWAP or reversal signal
            if close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals