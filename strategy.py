#!/usr/bin/env python3
# 4h_Trend_Pullback_MA50_100_Signal
# Hypothesis: In trending markets, price pulls back to the 50-period MA on 4h
# offers high-probability entry in direction of the 100-period MA trend.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)
# by using the 100-period MA as trend filter. Volume confirmation filters false signals.
# Target: 20-40 trades/year on 4h to avoid fee drag.

name = "4h_Trend_Pullback_MA50_100_Signal"
timeframe = "4h"
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
    
    # Get daily data for trend filter (100-period MA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate daily 100-period MA for trend filter
    ma_100_1d = pd.Series(df_1d['close']).rolling(window=100, min_periods=100).mean().values
    ma_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_100_1d)
    
    # Calculate 50-period MA on 4h for pullback entries
    ma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily MA100 (100), 4h MA50 (50), volume MA (20)
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ma_100_1d_aligned[i]) or 
            np.isnan(ma_50[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter: 100-period MA direction
        uptrend = close[i] > ma_100_1d_aligned[i]
        downtrend = close[i] < ma_100_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price pulls back to near 50 MA + volume
            # Allow 0.5% tolerance around MA50 for entry
            near_ma50 = abs(close[i] - ma_50[i]) / ma_50[i] < 0.005
            if uptrend and near_ma50 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price rallies to near 50 MA + volume
            elif downtrend and near_ma50 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price moves significantly above 50 MA
            if not uptrend or close[i] > ma_50[i] * 1.01:  # 1% above MA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price moves significantly below 50 MA
            if not downtrend or close[i] < ma_50[i] * 0.99:  # 1% below MA50
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals