#!/usr/bin/env python3
name = "6h_Chaikin_Money_Flow_Range_Extreme_1dTrend"
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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Chaikin Money Flow (CMF) on 6h data
    # CMF = sum(volume * ((close - low) - (high - close)) / (high - low)) / sum(volume) over period
    # Range: ((close - low) - (high - close)) = 2*close - high - low
    mf_multiplier = ((2 * close - high - low) / (high - low))
    # Handle division by zero when high == low
    mf_multiplier = np.where((high - low) == 0, 0, mf_multiplier)
    mf_volume = mf_multiplier * volume
    
    # 20-period CMF
    mf_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mf_sum / vol_sum, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(cmf[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CMF < -0.25 (extreme selling) + price above 1d EMA50 (bullish trend bias)
            if cmf[i] < -0.25 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF > 0.25 (extreme buying) + price below 1d EMA50 (bearish trend bias)
            elif cmf[i] > 0.25 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CMF > -0.05 (selling pressure exhausted)
            if cmf[i] > -0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CMF < 0.05 (buying pressure exhausted)
            if cmf[i] < 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals