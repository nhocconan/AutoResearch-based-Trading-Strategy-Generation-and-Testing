#!/usr/bin/env python3
# 1h_4h_1d_TrendFollowing_VolumeBreakout
# Hypothesis: On 1h, enter long when price breaks above 4h high with volume confirmation and 1d uptrend (close > EMA50).
# Enter short when price breaks below 4h low with volume confirmation and 1d downtrend (close < EMA50).
# Exit when price returns to 4h midpoint or trend reverses.
# Uses 4h for breakout levels, 1d for trend filter, 1h for entry timing and volume.
# Designed to work in both bull (follow 1d uptrend) and bear (follow 1d downtrend) markets.
# Target: 15-35 trades/year by requiring volume > 2x average and clear 4h breakout.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_TrendFollowing_VolumeBreakout"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for breakout levels (high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h: High and low for breakout levels ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # === 1d: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all HTF data to 1h
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4h)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h midpoint for exit: (high + low) / 2
    midpoint_4h = (high_4h_aligned + low_4h_aligned) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        high_4h_val = high_4h_aligned[i]
        low_4h_val = low_4h_aligned[i]
        midpoint_4h_val = midpoint_4h[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_4h_val) or np.isnan(low_4h_val) or 
            np.isnan(midpoint_4h_val) or np.isnan(ema_50_1d_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above 4h high with volume confirmation and 1d uptrend (close > EMA50)
            if (close_val > high_4h_val and  # Price broke above 4h high
                ema_50_1d_val > 0 and  # Valid EMA
                close_val > ema_50_1d_val and  # Uptrend filter: price above 1d EMA50
                vol_ratio_val > 2.0):  # Volume confirmation (>2x average)
                signals[i] = 0.20
                position = 1
            # Short: Breakdown below 4h low with volume confirmation and 1d downtrend (close < EMA50)
            elif (close_val < low_4h_val and  # Price broke below 4h low
                  ema_50_1d_val > 0 and  # Valid EMA
                  close_val < ema_50_1d_val and  # Downtrend filter: price below 1d EMA50
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to 4h midpoint or trend reverses (price < EMA50)
            if close_val < midpoint_4h_val or close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns to 4h midpoint or trend reverses (price > EMA50)
            if close_val > midpoint_4h_val or close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals