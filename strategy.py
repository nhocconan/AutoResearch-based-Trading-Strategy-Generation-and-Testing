#!/usr/bin/env python3
"""
6h_ChaikinMoneyFlow_Signal_Strategy_v1
Hypothesis: Use Chaikin Money Flow (CMF) on 6-hour bars with a 1-day trend filter.
- Long when CMF > +0.1 and 1-day EMA50 is upward (close > EMA50)
- Short when CMF < -0.1 and 1-day EMA50 is downward (close < EMA50)
- Exit when CMF crosses back toward zero (|CMF| < 0.05) or trend reverses
CMF measures institutional money flow, which tends to persist in trends.
The 1-day EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
Designed for low turnover (target: 12-30 trades/year) to minimize fee impact on 6h timeframe.
Works in both bull and bear markets by following the 1-day trend.
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
    
    # Calculate 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 with proper initialization
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])  # Simple average for first value
        for i in range(50, len(close_1d)):
            k = 2 / (50 + 1)
            ema50_1d[i] = close_1d[i] * k + ema50_1d[i-1] * (1 - k)
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Chaikin Money Flow (CMF) on 6h data
    # CMF = AD (Accumulation Distribution) 21-period sum / Volume 21-period sum
    # Where AD = [(Close - Low) - (High - Close)] / (High - Low) * Volume
    # Handle division by zero when high == low
    hl_range = high - low
    # Avoid division by zero - when hl_range == 0, use 0 for the CLV component
    clv = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    ad = clv * volume
    
    # Calculate 21-period sums for CMF
    ad_sum = np.full(n, np.nan)
    vol_sum = np.full(n, np.nan)
    
    for i in range(20, n):  # 21-period: indices i-20 to i inclusive
        ad_sum[i] = np.sum(ad[i-20:i+1])
        vol_sum[i] = np.sum(volume[i-20:i+1])
    
    # CMF = AD_sum / Vol_sum, handle division by zero
    cmf = np.full(n, np.nan)
    for i in range(20, n):
        if vol_sum[i] != 0:
            cmf[i] = ad_sum[i] / vol_sum[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are ready: need 21 for CMF and 50 for EMA
    start_idx = max(20, 49)  # 20 for CMF (0-indexed, need 21 bars), 49 for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not ready
        if (np.isnan(cmf[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: CMF > +0.1 (buying pressure) and uptrend (close > EMA50)
            if cmf[i] > 0.10 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: CMF < -0.1 (selling pressure) and downtrend (close < EMA50)
            elif cmf[i] < -0.10 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF falls below +0.05 (weakening buying pressure) OR trend turns down
            if cmf[i] < 0.05 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF rises above -0.05 (weakening selling pressure) OR trend turns up
            if cmf[i] > -0.05 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ChaikinMoneyFlow_Signal_Strategy_v1"
timeframe = "6h"
leverage = 1.0