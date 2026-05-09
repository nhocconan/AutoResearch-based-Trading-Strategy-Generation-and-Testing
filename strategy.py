#!/usr/bin/env python3
# 6h_OrderFlow_Imbalance_Reversal
# Hypothesis: At extreme 6h price levels (3σ from 50-period mean), fading the move with volume confirmation works due to mean reversion in overextended moves.
# Uses z-score of price deviation from mean + volume spike to identify exhaustion. Works in both bull/bear as extremes get faded.
# Target: 20-40 trades/year on 6h timeframe.

name = "6h_OrderFlow_Imbalance_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Price deviation from 50-period mean
    price_mean = np.full_like(close, np.nan)
    price_std = np.full_like(close, np.nan)
    if len(close) >= 50:
        for i in range(49, len(close)):
            window = close[i-49:i+1]
            price_mean[i] = np.mean(window)
            price_std[i] = np.std(window)
    
    # Z-score of price deviation (how many std dev from mean)
    price_zscore = np.full_like(close, np.nan)
    valid = (~np.isnan(price_mean)) & (~np.isnan(price_std)) & (price_std > 0)
    price_zscore[valid] = (close[valid] - price_mean[valid]) / price_std[valid]
    
    # Volume spike detector (2x 20-period average)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma > 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Extreme conditions: price > 2 std dev above/below mean with volume confirmation
    extreme_long = (price_zscore < -2.0) & volume_ratio > 1.5  # Oversold with volume
    extreme_short = (price_zscore > 2.0) & volume_ratio > 1.5  # Overbought with volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(price_zscore[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long at extreme oversold with volume
            if extreme_long[i]:
                signals[i] = 0.25
                position = 1
            # Enter short at extreme overbought with volume
            elif extreme_short[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to mean or extreme reverses
            if price_zscore[i] > -0.5 or price_zscore[i] < -3.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to mean or extreme reverses
            if price_zscore[i] < 0.5 or price_zscore[i] > 3.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals