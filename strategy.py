#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1-week Bollinger Band squeeze breakout with volume confirmation.
In both bull and bear markets, Bollinger Band squeezes (low volatility) precede explosive moves.
Combines with 1d volume spike and breakout direction for entry.
Uses 1w for volatility regime (lower frequency) and 1d for entry timing to reduce overtrading.
Target: 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for Bollinger Band squeeze
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band squeeze: width below 20-period average width
    avg_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < avg_width
    
    # Squeeze release: width expanding after squeeze
    squeeze_release = np.zeros_like(squeeze, dtype=bool)
    for i in range(1, len(squeeze)):
        if not squeeze[i-1] and squeeze[i]:
            squeeze_release[i] = True  # Just entered squeeze
        elif squeeze[i-1] and not squeeze[i]:
            squeeze_release[i] = True  # Just exited squeeze (breakout)
    
    squeeze_release_aligned = align_htf_to_ltf(prices, df_1w, squeeze_release.astype(float))
    
    # 1d volume confirmation: volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # 1d price breakout direction: close > upper BB or close < lower BB (1d)
    sma_20_1d = pd.Series(prices['close'].values).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(prices['close'].values).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    
    breakout_up = prices['close'].values > upper_bb_1d
    breakout_down = prices['close'].values < lower_bb_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(squeeze_release_aligned[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(breakout_up[i]) or 
            np.isnan(breakout_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: BB squeeze release + volume spike + upward breakout
            if (squeeze_release_aligned[i] > 0.5 and 
                vol_ratio[i] > 2.0 and 
                breakout_up[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: BB squeeze release + volume spike + downward breakout
            elif (squeeze_release_aligned[i] > 0.5 and 
                  vol_ratio[i] > 2.0 and 
                  breakout_down[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: volatility expansion ends or opposite breakout
            vol_ratio_threshold = 1.2
            if position == 1 and (vol_ratio[i] < vol_ratio_threshold or breakout_down[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (vol_ratio[i] < vol_ratio_threshold or breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_BollingerSqueeze_Breakout_Volume"
timeframe = "1d"
leverage = 1.0