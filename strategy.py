#!/usr/bin/env python3
name = "6h_WeeklyPivotBias_VolumeSpike_1dTrend"
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
    
    # Weekly pivot bias (from weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly trend: price above/below 20-week SMA
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    weekly_uptrend = close > sma_20_1w_aligned
    
    # Daily volume spike (volume > 1.5x 20-day average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma20
    
    # 6h price action: close above/below 20-period SMA for trend confirmation
    sma_20_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # ensure indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(weekly_uptrend[i]) or np.isnan(volume_spike[i]) or np.isnan(sma_20_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend, volume spike, price above 6h SMA
            if weekly_uptrend[i] and volume_spike[i] and close[i] > sma_20_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend, volume spike, price below 6h SMA
            elif not weekly_uptrend[i] and volume_spike[i] and close[i] < sma_20_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly downtrend or loss of volume spike or price below SMA
            if not weekly_uptrend[i] or not volume_spike[i] or close[i] < sma_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly uptrend or loss of volume spike or price above SMA
            if weekly_uptrend[i] or not volume_spike[i] or close[i] > sma_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals