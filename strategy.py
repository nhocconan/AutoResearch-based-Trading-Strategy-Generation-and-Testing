#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Bollinger Band squeeze breakout with volume confirmation.
In both bull and bear markets, low volatility (BB squeeze) precedes explosive moves.
Breakouts from squeeze with volume confirmation capture momentum.
Uses 1w for volatility regime (lower frequency) and 1d for entry timing.
Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for Bollinger Band squeeze
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band squeeze: low volatility regime
    # Squeeze when BB width is at 20-period low
    bb_width_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_20  # True when in squeeze
    
    # Breakout detection: price breaks above upper BB or below lower BB
    breakout_up = close_1w > upper_bb
    breakout_down = close_1w < lower_bb
    
    # Align to 1d timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze.astype(float))
    breakout_up_aligned = align_htf_to_ltf(prices, df_1w, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_1w, breakout_down.astype(float))
    
    # 1d volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 1w squeeze + 1w breakout up + volume confirmation
            if (squeeze_aligned[i] > 0.5 and 
                breakout_up_aligned[i] > 0.5 and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: 1w squeeze + 1w breakout down + volume confirmation
            elif (squeeze_aligned[i] > 0.5 and 
                  breakout_down_aligned[i] > 0.5 and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: squeeze ends or opposite breakout
            if position == 1:
                if squeeze_aligned[i] < 0.5 or breakout_down_aligned[i] > 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if squeeze_aligned[i] < 0.5 or breakout_up_aligned[i] > 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_BB_Squeeze_Breakout_Volume"
timeframe = "1d"
leverage = 1.0