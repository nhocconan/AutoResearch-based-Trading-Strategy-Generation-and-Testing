# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action filtered by 1d Bollinger Band squeeze and breakout.
# Long when price breaks above upper BB(20,2) AND BB width is at 20-day low (squeeze breakout).
# Short when price breaks below lower BB(20,2) AND BB width is at 20-day low.
# Exit when price returns to the middle BB (20-period SMA).
# Uses Bollinger Band squeeze as a volatility contraction/expansion filter to capture breakouts
# after low volatility periods, which works in both trending and ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "6h_Bollinger_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Bollinger Bands
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    close_d = df_d['close'].values
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    
    # Bollinger Band Width (normalized)
    bb_width = (upper_bb - lower_bb) / sma
    # 20-period minimum of BB width (squeeze condition)
    bb_width_min = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    squeeze = bb_width <= bb_width_min  # True when at or near 20-day low width
    
    # Align BB levels and squeeze to 6h timeframe
    upper_bb_6h = align_htf_to_ltf(prices, df_d, upper_bb)
    lower_bb_6h = align_htf_to_ltf(prices, df_d, lower_bb)
    squeeze_6h = align_htf_to_ltf(prices, df_d, squeeze)
    middle_bb_6h = align_htf_to_ltf(prices, df_d, sma)  # For exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_6h[i]) or np.isnan(lower_bb_6h[i]) or 
            np.isnan(squeeze_6h[i]) or np.isnan(middle_bb_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper BB during squeeze
            long_cond = (close[i] > upper_bb_6h[i]) and squeeze_6h[i]
            # Short: breakout below lower BB during squeeze
            short_cond = (close[i] < lower_bb_6h[i]) and squeeze_6h[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle BB
            if close[i] <= middle_bb_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle BB
            if close[i] >= middle_bb_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals