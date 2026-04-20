#!/usr/bin/env python3
"""
4h_1d_RangeBreakout_Volume_Trend_v1
Concept: 4h breakout from daily range (high/low) with volume confirmation and 1d trend filter.
- Long: Price > daily high AND volume > 1.5x volume MA(20) AND close > open on 1d (bullish day)
- Short: Price < daily low AND volume > 1.5x volume MA(20) AND close < open on 1d (bearish day)
- Exit: Price crosses back into daily range (between daily low and high)
- Position sizing: 0.25
- Target: 20-50 trades/year (80-200 total over 4 years)
Works in bull/bear: Daily range defines structure, volume confirms breakout, 1d trend filters counter-trend noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RangeBreakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # === 4h: Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily: Range and trend ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    
    # Daily trend: bullish if close > open, bearish if close < open
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    
    # Align daily data to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Get values
        high_1d_val = high_1d_aligned[i]
        low_1d_val = low_1d_aligned[i]
        bullish_val = daily_bullish_aligned[i]
        bearish_val = daily_bearish_aligned[i]
        vol_ma20_val = vol_ma20[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_1d_val) or np.isnan(low_1d_val) or 
            np.isnan(bullish_val) or np.isnan(bearish_val) or 
            np.isnan(vol_ma20_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above daily high with volume confirmation and bullish day
            if (prices['close'].iloc[i] > high_1d_val and 
                volume[i] > 1.5 * vol_ma20_val and bullish_val > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Break below daily low with volume confirmation and bearish day
            elif (prices['close'].iloc[i] < low_1d_val and 
                  volume[i] > 1.5 * vol_ma20_val and bearish_val > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks back below daily low
            if prices['close'].iloc[i] < low_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks back above daily high
            if prices['close'].iloc[i] > high_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals