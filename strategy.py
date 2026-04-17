#!/usr/bin/env python3
"""
12h_1dWMA_VolumeBreakout
- Uses 1d Weighted Moving Average (WMA) as trend filter
- Enters long when price breaks above WMA with volume > 2x 20-period average
- Enters short when price breaks below WMA with volume > 2x 20-period average
- Exit when price crosses back across WMA
- Position size: 0.25 to manage drawdown and limit trade frequency
- Designed for 12h timeframe targeting 50-150 total trades over 4 years
- Works in bull/bear via volume-confirmed breakouts aligned with daily trend
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
    
    # Get daily data for WMA trend filter
    df_1d = get_htf_data(prices, '1d')
    pclose = df_1d['close'].values
    
    # Calculate 1d WMA (20-period) for trend filter
    # WMA = sum(price * weight) / sum(weights), weights = 1..20
    weights = np.arange(1, 21)
    wma_20 = np.convolve(pclose, weights[::-1], mode='full')[:len(pclose)] @ weights / weights.sum()
    # Pad beginning with NaN for insufficient data
    wma_20 = np.concatenate([np.full(19, np.nan), wma_20[19:]])
    
    # Align daily WMA to 12h timeframe (waits for daily bar to close)
    wma_20_12h = align_htf_to_ltf(prices, df_1d, wma_20)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(wma_20_12h[i]) or np.isnan(volume_ma_20.iloc[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price crosses above WMA with volume spike
            if price > wma_20_12h[i] and close[i-1] <= wma_20_12h[i-1] and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below WMA with volume spike
            elif price < wma_20_12h[i] and close[i-1] >= wma_20_12h[i-1] and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below WMA
            if price < wma_20_12h[i] and close[i-1] >= wma_20_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above WMA
            if price > wma_20_12h[i] and close[i-1] <= wma_20_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dWMA_VolumeBreakout"
timeframe = "12h"
leverage = 1.0