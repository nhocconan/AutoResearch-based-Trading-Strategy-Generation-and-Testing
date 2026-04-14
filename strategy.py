#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        multiplier = 2 / (20 + 1)
        ema20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema20_1w[i] = (close_1w[i] - ema20_1w[i-1]) * multiplier + ema20_1w[i-1]
    
    # Align weekly EMA to 4h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate 20-period ATR (4h) for volatility and stop loss
    tr = np.zeros_like(high)
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i],
                   abs(high[i] - high[i-1]),
                   abs(low[i] - low[i-1]))
    
    atr = np.full_like(high, np.nan)
    if len(high) >= 20:
        atr[19] = np.mean(tr[1:20])
        for i in range(20, len(high)):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size: 25% of capital
    
    # Pre-calculate volume ratio for efficiency
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: price above weekly EMA with volume surge
            if (close[i] > ema20_1w_aligned[i] and 
                volume_ratio > 3.0):
                position = 1
                signals[i] = position_size
            # Short: price below weekly EMA with volume surge
            elif (close[i] < ema20_1w_aligned[i] and 
                  volume_ratio > 3.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA or volatility contraction
            if (close[i] < ema20_1w_aligned[i] or
                atr[i] < np.mean(atr[max(0, i-19):i+1]) * 0.7):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly EMA or volatility contraction
            if (close[i] > ema20_1w_aligned[i] or
                atr[i] < np.mean(atr[max(0, i-19):i+1]) * 0.7):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WeeklyEMA_Volume_Surge"
timeframe = "4h"
leverage = 1.0