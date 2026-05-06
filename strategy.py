#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ATR expansion + weekly Donchian breakout
# Long when 1d ATR(14) > 1.5 * ATR(50) (volatility expansion) AND price breaks above 1w Donchian(20) high AND volume > 1.5 * avg_volume(20) on 6h
# Short when 1d ATR(14) > 1.5 * ATR(50) (volatility expansion) AND price breaks below 1w Donchian(20) low AND volume > 1.5 * avg_volume(20) on 6h
# Exit when price returns to 1w Donchian(10) midpoint (mean reversion to center)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Volatility expansion identifies genuine breakouts vs false moves
# Weekly Donchian provides structural support/resistance from higher timeframe
# Volume confirmation validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "6h_1dATR_Expansion_1wDonchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed 1d bars for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = 0  # First bar: no previous close
    tr3[0] = 0  # First bar: no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) and ATR(50) for 1d
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR values to 6h timeframe (wait for completed 1d bar)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Get 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian(20) channels
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    highest_high_10 = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    
    # Align 1w Donchian values to 6h timeframe (wait for completed 1w bar)
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    highest_high_10_aligned = align_htf_to_ltf(prices, df_1w, highest_high_10)
    lowest_low_10_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_10)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or 
            np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(highest_high_10_aligned[i]) or np.isnan(lowest_low_10_aligned[i]) or
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ATR expansion + break above 1w Donchian(20) high + volume spike, in session
            if (atr_14_1d_aligned[i] > 1.5 * atr_50_1d_aligned[i] and 
                close[i] > highest_high_20_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: ATR expansion + break below 1w Donchian(20) low + volume spike, in session
            elif (atr_14_1d_aligned[i] > 1.5 * atr_50_1d_aligned[i] and 
                  close[i] < lowest_low_20_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1w Donchian(10) midpoint
            midpoint_10 = (highest_high_10_aligned[i] + lowest_low_10_aligned[i]) / 2.0
            if close[i] < midpoint_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1w Donchian(10) midpoint
            midpoint_10 = (highest_high_10_aligned[i] + lowest_low_10_aligned[i]) / 2.0
            if close[i] > midpoint_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals