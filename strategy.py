#!/usr/bin/env python3
"""
6h_ChaikinMoneyFlow_Trend_Confirmation
Hypothesis: Chaikin Money Flow (CMF) with 20-period window measures institutional accumulation/distribution.
When CMF crosses above +0.15 with 1-day EMA50 uptrend, go long; when CMF crosses below -0.15 with 1-day EMA50 downtrend, go short.
Exit when CMF returns to neutral zone (-0.05 to +0.05). Designed for 6h to achieve 15-35 trades/year with clear trend following.
Works in bull markets via accumulation signals and in bear markets via distribution signals, both confirmed by higher timeframe trend.
"""
name = "6h_ChaikinMoneyFlow_Trend_Confirmation"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    high_low = high - low
    # Avoid division by zero
    high_low_safe = np.where(high_low == 0, 1e-10, high_low)
    mfm = ((close - low) - (high - close)) / high_low_safe
    mfv = mfm * volume
    
    # 20-period sums
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(volume_sum == 0, 0, mfv_sum / volume_sum)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for CMF calculation
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(cmf[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF crosses above +0.15 + 1-day uptrend
            if cmf[i] > 0.15 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF crosses below -0.15 + 1-day downtrend
            elif cmf[i] < -0.15 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: CMF returns to neutral zone (-0.05 to +0.05)
            if -0.05 <= cmf[i] <= 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals